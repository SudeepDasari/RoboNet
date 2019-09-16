from .base_graph import BaseGraph
import itertools
import tensorflow as tf
import tensorflow.keras.layers as layers
from robonet.video_prediction.layers.dnaflow_rnn_cell import RELU_SHIFT
from robonet.video_prediction.ops import pad2d


def apply_cdna_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 4-D of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    batch_size, height, width, color_channels = image.get_shape().as_list()
    batch_size, kernel_height, kernel_width, num_transformed_images = kernels.get_shape().as_list()
    kernel_size = [kernel_height, kernel_width]
    image_padded = pad2d(image, kernel_size, rate=dilation_rate, padding='SAME', mode='CONSTANT')
    # Treat the color channel dimension as the batch dimension since the same
    # transformation is applied to each color channel.
    # Treat the batch dimension as the channel dimension so that
    # depthwise_conv2d can apply a different transformation to each sample.
    kernels = tf.transpose(kernels, [1, 2, 0, 3])
    kernels = tf.reshape(kernels, [kernel_size[0], kernel_size[1], batch_size, num_transformed_images])
    # Swap the batch and channel dimensions.
    image_transposed = tf.transpose(image_padded, [3, 1, 2, 0])
    # Transform image.
    outputs = tf.nn.depthwise_conv2d(image_transposed, kernels, [1, 1, 1, 1], padding='VALID', rate=dilation_rate)
    # Transpose the dimensions to where they belong.
    outputs = tf.reshape(outputs, [color_channels, height, width, batch_size, num_transformed_images])
    outputs = tf.transpose(outputs, [3, 1, 2, 4, 0])

    return outputs


def _cast_down(tensor, hparams):
    if hparams.float16 and hparams.use_tpu:
        return tf.cast(tensor, tf.bfloat16)
    elif hparams.float16:
        return tf.cast(tensor, tf.float16)
    return tensor


def _cast_up(tensor):
    if tensor.dtype != tf.float32:
        return tf.cast(tensor, tf.float32)
    return tensor


class VGGConvGraph(BaseGraph):
    def build_graph(self, mode, inputs, hparams, n_gpus=1, scope_name='generator'):
        if 'pix_distribs' in inputs:
            assert hparams.use_flows, "pixel distributions can only be used in flow mode!"
        
        # calculate number of flows if needed
        if hparams.use_flows:
            self._n_flows = (hparams.skip_flows * hparams.context_frames) + hparams.img_flows

        self._scope_name = scope_name
        #TODO "implement state conditioning"
        assert not hparams.use_states

        enc_device = dec_device = '/device:GPU:0'
        if n_gpus > 1:
            dec_device = '/device:GPU:1'
        print('encoder on device {} and decoder on device {}'.format(enc_device, dec_device))

        with tf.variable_scope(self._scope_name) as graph_scope:
            self._init_layers(hparams, inputs, mode, enc_device, dec_device)
            T, B, H, W, C = inputs['images'].get_shape().as_list()

            enc_lstm_state, dec_lstm_state = None, None
            previous_encs = []
            outputs = {}

            for t in range(hparams.sequence_length - 1):
                norm_ctr = 0
                print('building graph for t={}'.format(t), end="\r")
                action_state_vector = _cast_down(inputs['actions'][t], hparams)
                if t < hparams.context_frames:
                    input_image = _cast_down(inputs['images'][t], hparams)
                else:
                    casted_real = _cast_down(inputs['images'][t], hparams)
                    casted_gen = _cast_down(outputs['gen_images'][-1], hparams)
                    input_image = tf.where(self._ground_truth[t], casted_real, casted_gen)

                with tf.device(enc_device):
                    # encoder convs
                    encoded_imgs = input_image
                    for i, op in enumerate(self._enc_ops):
                        encoded_imgs = op(encoded_imgs)
                        if isinstance(op, layers.Conv2D):
                            encoded_imgs = _cast_up(encoded_imgs)
                            encoded_imgs = tf.contrib.layers.instance_norm(encoded_imgs, activation_fn=tf.nn.relu, 
                                                                            scope='norm{}'.format(norm_ctr), reuse = t > 0)
                            encoded_imgs = _cast_down(encoded_imgs, hparams)
                            norm_ctr += 1

                    # encode actions and append to hidden state
                    enc_append = tf.reshape(self._enc_append(action_state_vector), (B, int(H // 8), int(W // 8), 
                                            hparams.action_append_channels))
                    encoded_imgs = self._enc_conv(tf.concat((encoded_imgs, enc_append), -1))
                    encoded_imgs = tf.contrib.layers.instance_norm(_cast_up(encoded_imgs), activation_fn=tf.nn.relu, 
                                                                    scope='norm{}'.format(norm_ctr), reuse = t > 0)
                
                    norm_ctr += 1
                    # encoder lstm cell
                    if t == 0:
                        enc_lstm_state = self._enc_lstm.get_initial_state(encoded_imgs[:, None])
                    enc_out, enc_lstm_state = self._enc_lstm.cell(encoded_imgs, enc_lstm_state)

                with tf.device(dec_device):
                    # decoder attention
                    enc_out = _cast_down(enc_out, hparams)
                    flatten_enc = tf.reshape(enc_out, (B, -1))
                    if t == 0:
                        previous_encs = [flatten_enc]
                        attention_enc = enc_out
                    else:
                        previous_encs.append(flatten_enc)
                        dot_prods = [tf.reduce_sum(flatten_enc * x, 1, keepdims=True) for x in previous_encs]
                        attention_weights = tf.nn.softmax(tf.concat(dot_prods, axis=1), axis=1)
                        attention_enc = tf.reduce_sum(attention_weights[:, :, None] * tf.concat([p[:, None] for p in previous_encs], 1), 1)
                        attention_enc = tf.reshape(attention_enc, [B, int(H // 8), int(W // 8), hparams.lstm_filters])
                    attention_enc = _cast_up(attention_enc)

                    # decoder lstm cell
                    if t == 0:
                        dec_lstm_state = self._dec_lstm.get_initial_state(attention_enc[:, None])
                    dec_lstm_out, dec_lstm_state = self._dec_lstm.cell(attention_enc, dec_lstm_state)
                    
                    # decoder convs
                    if t < hparams.context_frames - 1:   # no frame predictions for extra context frames
                        continue
                
                    decoder_out = _cast_down(dec_lstm_out, hparams)
                    for op in self._dec_ops:
                        decoder_out = op(decoder_out)
                        if isinstance(op, layers.Conv2D):
                            decoder_out = _cast_up(decoder_out)
                            decoder_out = tf.contrib.layers.instance_norm(decoder_out, activation_fn=tf.nn.relu, 
                                                                            scope='norm{}'.format(norm_ctr), reuse = t >= hparams.context_frames)
                            decoder_out = _cast_down(decoder_out, hparams)
                            norm_ctr += 1

                    # predict images and pixel distributions using flows
                    if hparams.use_flows:
                        kernel_convs, mask_convs = tf.split(decoder_out, 2, axis=-1)
                        kernel_convs = tf.transpose(tf.reshape(kernel_convs, (B, -1, self._n_flows)), (0, 2, 1))
                        kernels = tf.nn.relu(self._kernel_top(kernel_convs - RELU_SHIFT)) + RELU_SHIFT
                        kernels = tf.transpose(kernels, (0, 2, 1))
                        kernels = tf.reshape(kernels / tf.reduce_sum(kernels, axis=1, keepdims=True), (B, hparams.cdna_kernel_size, -1, self._n_flows))

                        warped_images = []
                        if hparams.skip_flows:
                            warped_images = [apply_cdna_kernels(inputs['images'][t_index], 
                                                    kernels[:, :, :, t_index * hparams.skip_flows: (t_index + 1) * hparams.skip_flows])
                                                    for t_index in range(hparams.context_frames)]
                        img_flow_kernels = kernels[:, :, :, hparams.context_frames * hparams.skip_flows:]
                        warped_images.append(apply_cdna_kernels(input_image, img_flow_kernels))
                        warped_images = tf.concat(warped_images, axis=-2)

                        masks = tf.expand_dims(tf.nn.softmax(self._mask_top(mask_convs)), axis=-1)

                        outputs['gen_images'] = outputs.get('gen_images', []) + [_cast_up(tf.reduce_sum(warped_images * masks, axis=-2))]

                        if 'pix_distribs' in inputs:
                            warped_distribs = []
                            if hparams.skip_flows:
                                warped_distribs = [apply_cdna_kernels(inputs['pix_distribs'][t_index], 
                                                        kernels[:, :, :, t_index * hparams.skip_flows: (t_index + 1) * hparams.skip_flows])
                                                        for t_index in range(hparams.context_frames)]

                            warped_distribs.append(apply_cdna_kernels(inputs['pix_distribs'][t], img_flow_kernels))
                            warped_distribs = tf.concat(warped_distribs, axis=-2)
                            warped_distribs = _cast_up(tf.reduce_sum(warped_distribs * masks, axis=-2))
                            warped_distribs = warped_distribs / (tf.reduce_sum(warped_distribs, axis=(1, 2), keepdims=True) + RELU_SHIFT)
                            outputs['gen_pix_distribs'] = outputs.get('gen_pix_distribs', []) + [warped_distribs]
                    else:
                        outputs['gen_images'] = outputs.get('gen_images', []) + [_cast_up(self._top(decoder_out))]

            outputs['gen_images'] = tf.concat([pred[None] for pred in outputs['gen_images']], 0)
            outputs['ground_truth_sampling_mean'] = tf.reduce_mean(tf.to_float(self._ground_truth[hparams.context_frames:]))
            if 'pix_distribs' in inputs:
                outputs['gen_pix_distribs'] = tf.concat([pred[None] for pred in outputs['gen_pix_distribs']], 0)

        return outputs

    def _init_layers(self, hparams, inputs, mode, enc_device, dec_device):
        T, B, H, W, C = inputs['images'].get_shape().as_list()
        
        with tf.device(enc_device):
            self._conv_1_1 = layers.Conv2D(hparams.enc_filters[0], hparams.kernel_size, padding='same')
            self._conv_1_2 = layers.Conv2D(hparams.enc_filters[0], hparams.kernel_size, padding='same')
            self._pool_1 = layers.MaxPool2D()

            self._conv_2_1 = layers.Conv2D(hparams.enc_filters[1], hparams.kernel_size, padding='same')
            self._conv_2_2 = layers.Conv2D(hparams.enc_filters[1], hparams.kernel_size, padding='same')
            self._conv_2_3 = layers.Conv2D(hparams.enc_filters[1], hparams.kernel_size, padding='same')
            self._pool_2 = layers.MaxPool2D()

            self._conv_3_1 = layers.Conv2D(hparams.enc_filters[2], hparams.kernel_size, padding='same')
            self._conv_3_2 = layers.Conv2D(hparams.enc_filters[2], hparams.kernel_size, padding='same')
            self._conv_3_3 = layers.Conv2D(hparams.enc_filters[2], hparams.kernel_size, padding='same')
            self._conv_3_4 = layers.Conv2D(hparams.enc_filters[2], hparams.kernel_size, padding='same')
            self._pool_3 = layers.MaxPool2D()

            self._enc_ops = [self._conv_1_1, self._conv_1_2, self._pool_1, self._conv_2_1, self._conv_2_2, self._conv_2_3,
                            self._pool_2, self._conv_3_1, self._conv_3_2, self._conv_3_3, self._conv_3_4, self._pool_3]

            enc_H, enc_W = [int(x // 8) for x in (H, W)]
            self._enc_append = layers.Dense(enc_H * enc_W * hparams.action_append_channels)
            self._enc_conv = layers.Conv2D(hparams.lstm_filters, 1, padding='same')

            self._enc_lstm = layers.ConvLSTM2D(hparams.lstm_filters, hparams.kernel_size, padding = 'same')
            self._enc_lstm.cell.build([B, T, enc_H, enc_W, hparams.lstm_filters])

        with tf.device(dec_device):
            self._dec_lstm = layers.ConvLSTM2D(hparams.lstm_filters, hparams.kernel_size, padding='same')
            self._dec_lstm.cell.build([B, T, enc_H, enc_W, hparams.lstm_filters])

            self._conv_tranpose_1 = layers.Conv2DTranspose(hparams.lstm_filters, 2, strides=2)
            self._dec_conv_1_1 = layers.Conv2D(hparams.dec_filters[0], hparams.kernel_size, padding='same')
            self._dec_conv_1_2 = layers.Conv2D(hparams.dec_filters[0], hparams.kernel_size, padding='same')
            self._dec_conv_1_3 = layers.Conv2D(hparams.dec_filters[0], hparams.kernel_size, padding='same')
            self._dec_conv_1_4 = layers.Conv2D(hparams.dec_filters[0], hparams.kernel_size, padding='same')

            self._conv_tranpose_2 = layers.Conv2DTranspose(hparams.dec_filters[0], 2, strides=2)
            self._dec_conv_2_1 = layers.Conv2D(hparams.dec_filters[1], hparams.kernel_size, padding='same')
            self._dec_conv_2_2 = layers.Conv2D(hparams.dec_filters[1], hparams.kernel_size, padding='same')
            self._dec_conv_2_3 = layers.Conv2D(hparams.dec_filters[1], hparams.kernel_size, padding='same')

            self._conv_tranpose_3 = layers.Conv2DTranspose(hparams.dec_filters[1], 2, strides=2)
            self._dec_ops = [self._conv_tranpose_1, self._dec_conv_1_1, self._dec_conv_1_2, self._dec_conv_1_3, self._dec_conv_1_4,
                            self._conv_tranpose_2, self._dec_conv_2_1, self._dec_conv_2_2, self._dec_conv_2_3,
                            self._conv_tranpose_3]
            
            if hparams.use_flows:
                self._kernel_mask_conv = layers.Conv2D(self._n_flows * 2, 1, padding='same')
                self._dec_ops.append(self._kernel_mask_conv)

                # kernel prediction
                self._kernel_top = layers.Dense(hparams.cdna_kernel_size ** 2)
                # mask prediction
                self._mask_top = layers.Conv2D(self._n_flows, hparams.kernel_size, padding='same')
            else:
                self._dec_conv_3_1 = layers.Conv2D(3, hparams.kernel_size, padding='same')
                self._top = layers.Conv2D(3, hparams.kernel_size, padding='same')

                self._dec_ops.append(self._dec_conv_3_1)

        ground_truth_sampling_shape = [hparams.sequence_length - 1 - hparams.context_frames, B]
        if hparams.schedule_sampling == 'none' or mode != tf.estimator.ModeKeys.TRAIN:
            ground_truth_sampling = tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape)
        elif hparams.schedule_sampling in ('inverse_sigmoid', 'linear'):
            if hparams.schedule_sampling == 'inverse_sigmoid':
                k = hparams.schedule_sampling_k
                start_step = hparams.schedule_sampling_steps[0]
                iter_num = tf.to_float(tf.train.get_or_create_global_step())
                prob = (k / (k + tf.exp((iter_num - start_step) / k)))
                prob = tf.cond(tf.less(iter_num, start_step), lambda: 1.0, lambda: prob)
            elif hparams.schedule_sampling == 'linear':
                start_step, end_step = hparams.schedule_sampling_steps
                step = tf.clip_by_value(tf.train.get_or_create_global_step(), start_step, end_step)
                prob = 1.0 - tf.to_float(step - start_step) / tf.to_float(end_step - start_step)
            log_probs = tf.log([1 - prob, prob])
            ground_truth_sampling = tf.multinomial([log_probs] * B, ground_truth_sampling_shape[0])
            ground_truth_sampling = tf.cast(tf.transpose(ground_truth_sampling, [1, 0]), dtype=tf.bool)
            # Ensure that eventually, the model is deterministically
            # autoregressive (as opposed to autoregressive with very high probability).
            ground_truth_sampling = tf.cond(tf.less(prob, 0.001),
                                            lambda: tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape),
                                            lambda: ground_truth_sampling)
        else:
            raise NotImplementedError
        ground_truth_context = tf.constant(True, dtype=tf.bool, shape=[hparams.context_frames, B])
        self._ground_truth = tf.concat([ground_truth_context, ground_truth_sampling], axis=0)

    @staticmethod
    def default_hparams():
        default_params =  {
            "enc_filters": [32, 128, 256],
            "lstm_filters": 256,
            "dec_filters": [128, 64],
            "kernel_size": 3,
            'action_append_channels': 2,

            "use_flows": True,
            "img_flows": 12,
            "skip_flows": 1,
            "cdna_kernel_size": 10,

            'schedule_sampling': "inverse_sigmoid",
            'schedule_sampling_k': 900.0,
            'schedule_sampling_steps': [0, 100000],

            "float16": False                        # float 16 is very unstable at the moment
        }
        return dict(itertools.chain(BaseGraph.default_hparams().items(), default_params.items()))
