from .base_graph import BaseGraph
import itertools
import tensorflow as tf
import tensorflow.keras.layers as layers
from robonet.video_prediction.flow_ops import image_warp


class VGGConvGraph(BaseGraph):
    def build_graph(self, mode, inputs, hparams, scope_name='flow_generator'):
        self._scope_name = scope_name
        #TODO "implement state conditioning"
        assert not hparams.use_states

        with tf.variable_scope(self._scope_name) as graph_scope:
            self._init_layers(hparams, inputs, mode)
            T, B, H, W, C = inputs['images'].get_shape().as_list()

            enc_lstm_state, dec_lstm_state = None, None
            previous_encs = []
            outputs = {}

            for t in range(hparams.sequence_length - 1):
                norm_ctr = 0
                print('building graph for t={}'.format(t), end="\r")
                action_state_vector = inputs['actions'][t]
                if t < hparams.context_frames:
                    input_image = inputs['images'][t]
                else:
                    input_image = tf.where(self._ground_truth[t], inputs['images'][t], outputs['gen_images'][-1])

                # encoder convs
                encoded_imgs = input_image
                for i, op in enumerate(self._enc_ops):
                    encoded_imgs = op(encoded_imgs)
                    if isinstance(op, layers.Conv2D):
                        encoded_imgs = tf.contrib.layers.instance_norm(encoded_imgs, activation_fn=tf.nn.relu, 
                                                                        scope='norm{}'.format(norm_ctr), reuse = t > 0)
                        norm_ctr += 1

                # encode actions and append to hidden state
                enc_append = tf.reshape(self._enc_append(action_state_vector), (B, int(H // 8), int(W // 8), 
                                        hparams.action_append_channels))
                encoded_imgs = self._enc_conv(tf.concat((encoded_imgs, enc_append), -1))
                encoded_imgs = tf.contrib.layers.instance_norm(encoded_imgs, activation_fn=tf.nn.relu, 
                                                                scope='norm{}'.format(norm_ctr), reuse = t > 0)
                norm_ctr += 1

                # encoder lstm cell
                if t == 0:
                    enc_lstm_state = self._enc_lstm.get_initial_state(encoded_imgs[:, None])
                enc_out, enc_lstm_state = self._enc_lstm.cell(encoded_imgs, enc_lstm_state)

                # decoder attention
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
                
                # decoder lstm cell
                if t == 0:
                    dec_lstm_state = self._dec_lstm.get_initial_state(attention_enc[:, None])
                dec_lstm_out, dec_lstm_state = self._dec_lstm.cell(attention_enc, dec_lstm_state)
                
                # decoder convs
                if t < hparams.context_frames - 1:   # no frame predictions for extra context frames
                    continue
                
                decoder_out = dec_lstm_out
                for op in self._dec_ops:
                    decoder_out = op(decoder_out)
                    if isinstance(op, layers.Conv2D):
                        decoder_out = tf.contrib.layers.instance_norm(decoder_out, activation_fn=tf.nn.relu, 
                                                                        scope='norm{}'.format(norm_ctr), reuse = t >= hparams.context_frames)
                        norm_ctr += 1

                # predict flows
                if hparams.use_flows:
                    flows_masks = tf.reshape(self._top(decoder_out), (B, H, W, hparams.n_flows, 3))
                    flows, masks = flows_masks[:, :, :, :, :2], flows_masks[:, :, :, :, -1]
                    masks = tf.expand_dims(tf.nn.softmax(masks), -1)

                    pred_imgs = []
                    if hparams.n_flows > hparams.context_frames:
                        source_imgs = [inputs['images'][fr_i] for fr_i in range(hparams.context_frames)] + [input_image for _ in range(hparams.n_flows - hparams.context_frames)]
                    else:
                        source_imgs = [input_image for _ in range(hparams.n_flows)]

                    for f, source_img in enumerate(source_imgs):
                        warped_image = tf.contrib.image.dense_image_warp(source_img, flows[:, :, :, f])
                        pred_imgs.append(warped_image * masks[:, :, :, f])

                    outputs['gen_images'] = outputs.get('gen_images', []) + [tf.math.accumulate_n(pred_imgs)]
                    outputs['gen_flows'] = outputs.get('gen_flows', []) + [tf.transpose(flows, [0, 1, 2, 4, 3])]
                else:
                    outputs['gen_images'] = outputs.get('gen_images', []) + [self._top(decoder_out)]

            if hparams.use_flows:
                outputs['gen_flows'] = tf.concat([flow[None] for flow in outputs['gen_flows']], 0)
            outputs['gen_images'] = tf.concat([pred[None] for pred in outputs['gen_images']], 0)
            outputs['ground_truth_sampling_mean'] = tf.reduce_mean(tf.to_float(self._ground_truth[hparams.context_frames:]))
        return outputs

    def _init_layers(self, hparams, inputs, mode):
        T, B, H, W, C = inputs['images'].get_shape().as_list()

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

        if hparams.use_flows:
            self._conv_tranpose_3 = layers.Conv2DTranspose(hparams.dec_filters[1], 2, strides=2)
            self._dec_conv_3_1 = layers.Conv2D(hparams.n_flows * 3, hparams.kernel_size, padding='same')
            self._top = layers.Conv2D(hparams.n_flows * 3, hparams.kernel_size, padding='same')
        else:
            self._conv_tranpose_3 = layers.Conv2DTranspose(hparams.dec_filters[1], 2, strides=2)
            self._dec_conv_3_1 = layers.Conv2D(3, hparams.kernel_size, padding='same')
            self._top = layers.Conv2D(3, hparams.kernel_size, padding='same')

        self._dec_ops = [self._conv_tranpose_1, self._dec_conv_1_1, self._dec_conv_1_2, self._dec_conv_1_3, self._dec_conv_1_4,
                        self._conv_tranpose_2, self._dec_conv_2_1, self._dec_conv_2_2, self._dec_conv_2_3,
                        self._conv_tranpose_3, self._dec_conv_3_1]

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

    @property
    def vars(self):
        return tf.trainable_variables(self._scope_name)

    @staticmethod
    def default_hparams():
        default_params =  {
            "enc_filters": [32, 128, 256],
            "lstm_filters": 256,
            "dec_filters": [128, 64],
            "kernel_size": 3,
            'action_append_channels': 2,

            "use_flows": True,
            "n_flows": 12,

            'schedule_sampling': "inverse_sigmoid",
            'schedule_sampling_k': 900.0,
            'schedule_sampling_steps': [0, 100000],
        }
        return dict(itertools.chain(BaseGraph.default_hparams().items(), default_params.items()))
