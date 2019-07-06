import tensorflow as tf
from robonet.video_prediction import ops, flow_ops
from robonet.video_prediction.ops import dense, conv2d, flatten, tile_concat
from robonet.video_prediction.rnn_ops import BasicConv2DLSTMCell, Conv2DGRUCell
from robonet.video_prediction.utils import tf_utils

import numpy as np

RELU_SHIFT = 1e-7


def apply_dna_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 6-D of shape
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    dilation_rate = list(dilation_rate) if isinstance(dilation_rate, (tuple, list)) else [dilation_rate] * 2
    batch_size, height, width, color_channels = image.get_shape().as_list()
    batch_size, height, width, kernel_height, kernel_width, num_transformed_images = kernels.get_shape().as_list()
    kernel_size = [kernel_height, kernel_width]

    # Flatten the spatial dimensions.
    kernels_reshaped = tf.reshape(kernels, [batch_size, height, width,
                                            kernel_size[0] * kernel_size[1], num_transformed_images])
    image_padded = pad2d(image, kernel_size, rate=dilation_rate, padding='SAME', mode='SYMMETRIC')
    # Combine channel and batch dimensions into the first dimension.
    image_transposed = tf.transpose(image_padded, [3, 0, 1, 2])
    image_reshaped = flatten(image_transposed, 0, 1)[..., None]
    patches_reshaped = tf.extract_image_patches(image_reshaped, ksizes=[1] + kernel_size + [1],
                                                strides=[1] * 4, rates=[1] + dilation_rate + [1], padding='VALID')
    # Separate channel and batch dimensions, and move channel dimension.
    patches_transposed = tf.reshape(patches_reshaped, [color_channels, batch_size, height, width, kernel_size[0] * kernel_size[1]])
    patches = tf.transpose(patches_transposed, [1, 2, 3, 0, 4])
    # Reduce along the spatial dimensions of the kernel.
    outputs = tf.matmul(patches, kernels_reshaped)
    outputs = tf.unstack(outputs, axis=-1)
    return outputs


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
    image_padded = pad2d(image, kernel_size, rate=dilation_rate, padding='SAME', mode='SYMMETRIC')
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
    outputs = tf.transpose(outputs, [4, 3, 1, 2, 0])
    outputs = tf.unstack(outputs, axis=0)
    return outputs


def apply_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 4-D or 6-D tensor of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]` or
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    if isinstance(image, list):
        image_list = image
        kernels_list = tf.split(kernels, len(image_list), axis=-1)
        outputs = []
        for image, kernels in zip(image_list, kernels_list):
            outputs.extend(apply_kernels(image, kernels))
    else:
        if len(kernels.get_shape()) == 4:
            outputs = apply_cdna_kernels(image, kernels, dilation_rate=dilation_rate)
        elif len(kernels.get_shape()) == 6:
            outputs = apply_dna_kernels(image, kernels, dilation_rate=dilation_rate)
        else:
            raise ValueError
    return outputs


def apply_flows(image, flows):
    if isinstance(image, list):
        image_list = image
        flows_list = tf.split(flows, len(image_list), axis=-1)
        outputs = []
        for image, flows in zip(image_list, flows_list):
            outputs.extend(apply_flows(image, flows))
    else:
        flows = tf.unstack(flows, axis=-1)
        outputs = [flow_ops.image_warp(image, flow) for flow in flows]
    return outputs


def identity_kernel(kernel_size):
    kh, kw = kernel_size
    kernel = np.zeros(kernel_size)

    def center_slice(k):
        if k % 2 == 0:
            return slice(k // 2 - 1, k // 2 + 1)
        else:
            return slice(k // 2, k // 2 + 1)

    kernel[center_slice(kh), center_slice(kw)] = 1.0
    kernel /= np.sum(kernel)
    return kernel


class VPredCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, mode, inputs, hparams, reuse=None):
        super(VPredCell, self).__init__(_reuse=reuse)
        self._mode = mode
        self.inputs = inputs
        self.hparams = hparams

        if self.hparams.where_add not in ('input', 'all', 'middle'):
            raise ValueError('Invalid where_add %s' % self.hparams.where_add)
        
        assert len(self.hparams.encoder_layer_size_mult) == len(self.hparams.encoder_layer_use_rnn)
        assert len(self.hparams.decoder_layer_size_mult) == len(self.hparams.decoder_layer_use_rnn)

        self.encoder_layer_specs = [(mult * self.hparams.ngf, use_rnn) for mult, use_rnn in 
                                    zip(self.hparams.encoder_layer_size_mult, self.hparams.encoder_layer_use_rnn)]
        self.decoder_layer_specs = [(mult * self.hparams.ngf, use_rnn) for mult, use_rnn in 
                                    zip(self.hparams.decoder_layer_size_mult, self.hparams.decoder_layer_use_rnn)]

        batch_size = inputs['images'].shape[1].value
        image_shape = inputs['images'].shape.as_list()[2:]
        height, width, _ = image_shape

        # output_size
        gen_input_shape = list(image_shape)
        if 'actions' in inputs:
            gen_input_shape[-1] += inputs['actions'].shape[-1].value
        num_masks = self.hparams.last_frames * self.hparams.num_transformed_images + \
            int(bool(self.hparams.prev_image_background)) + \
            int(bool(self.hparams.first_image_background and not self.hparams.context_images_background)) + \
            (self.hparams.context_frames if self.hparams.context_images_background else 0) + \
            int(bool(self.hparams.generate_scratch_image))
        output_size = {
            'gen_images': tf.TensorShape(image_shape),
            'gen_inputs': tf.TensorShape(gen_input_shape),
            'transformed_images': tf.TensorShape(image_shape + [num_masks]),
            'masks': tf.TensorShape([height, width, 1, num_masks]),
        }
        if 'pix_distribs' in inputs:
            num_motions = inputs['pix_distribs'].shape[-1].value
            output_size['gen_pix_distribs'] = tf.TensorShape([height, width, num_motions])
            output_size['transformed_pix_distribs'] = tf.TensorShape([height, width, num_motions, num_masks])
        if 'states' in inputs:
            output_size['gen_states'] = inputs['states'].shape[2:]
        if 'zrs' in inputs:
            output_size['zat_mu'] = self.hparams.za_dim
            output_size['zat_log_sigma_sq'] = self.hparams.za_dim
            output_size['gen_actions'] = inputs['actions'].shape[-1].value
        if 'zr_mu' in inputs and 'zr_log_sigma_sq' in inputs:
            output_size['zr_mu'] = self.hparams.zr_dim
            output_size['zr_log_sigma_sq'] = self.hparams.zr_dim
        if 'e' in inputs:
            output_size['e'] = self.hparams.e_dim

        if self.hparams.transformation == 'flow':
            output_size['gen_flows'] = tf.TensorShape([height, width, 2, self.hparams.last_frames * self.hparams.num_transformed_images])
            output_size['gen_flows_rgb'] = tf.TensorShape([height, width, 3, self.hparams.last_frames * self.hparams.num_transformed_images])
        self._output_size = output_size

        # state_size
        conv_rnn_state_sizes = []
        conv_rnn_height, conv_rnn_width = height, width
        for out_channels, use_conv_rnn in self.encoder_layer_specs:
            conv_rnn_height //= 2
            conv_rnn_width //= 2
            if use_conv_rnn:
                conv_rnn_state_sizes.append(tf.TensorShape([conv_rnn_height, conv_rnn_width, out_channels]))
        for out_channels, use_conv_rnn in self.decoder_layer_specs:
            conv_rnn_height *= 2
            conv_rnn_width *= 2
            if use_conv_rnn:
                conv_rnn_state_sizes.append(tf.TensorShape([conv_rnn_height, conv_rnn_width, out_channels]))
        if self.hparams.conv_rnn == 'lstm':
            conv_rnn_state_sizes = [tf.nn.rnn_cell.LSTMStateTuple(conv_rnn_state_size, conv_rnn_state_size)
                                    for conv_rnn_state_size in conv_rnn_state_sizes]
        state_size = {'time': tf.TensorShape([]),
                      'gen_image': tf.TensorShape(image_shape),
                      'last_images': [tf.TensorShape(image_shape)] * self.hparams.last_frames,
                      'conv_rnn_states': conv_rnn_state_sizes}
        if 'pix_distribs' in inputs:
            state_size['gen_pix_distrib'] = tf.TensorShape([height, width, num_motions])
            state_size['last_pix_distribs'] = [tf.TensorShape([height, width, num_motions])] * self.hparams.last_frames
        if 'states' in inputs:
            state_size['gen_state'] = inputs['states'].shape[2:]
        self._state_size = state_size

        ground_truth_sampling_shape = [self.hparams.sequence_length - 1 - self.hparams.context_frames, batch_size]
        if self.hparams.schedule_sampling == 'none' or self._mode != tf.esimator.ModeKeys.TRAIN:
            ground_truth_sampling = tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape)
        elif self.hparams.schedule_sampling in ('inverse_sigmoid', 'linear'):
            if self.hparams.schedule_sampling == 'inverse_sigmoid':
                k = self.hparams.schedule_sampling_k
                start_step = self.hparams.schedule_sampling_steps[0]
                iter_num = tf.to_float(tf.train.get_or_create_global_step())
                prob = (k / (k + tf.exp((iter_num - start_step) / k)))
                prob = tf.cond(tf.less(iter_num, start_step), lambda: 1.0, lambda: prob)
            elif self.hparams.schedule_sampling == 'linear':
                start_step, end_step = self.hparams.schedule_sampling_steps
                step = tf.clip_by_value(tf.train.get_or_create_global_step(), start_step, end_step)
                prob = 1.0 - tf.to_float(step - start_step) / tf.to_float(end_step - start_step)
            log_probs = tf.log([1 - prob, prob])
            ground_truth_sampling = tf.multinomial([log_probs] * batch_size, ground_truth_sampling_shape[0])
            ground_truth_sampling = tf.cast(tf.transpose(ground_truth_sampling, [1, 0]), dtype=tf.bool)
            # Ensure that eventually, the model is deterministically
            # autoregressive (as opposed to autoregressive with very high probability).
            ground_truth_sampling = tf.cond(tf.less(prob, 0.001),
                                            lambda: tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape),
                                            lambda: ground_truth_sampling)
        else:
            raise NotImplementedError
        ground_truth_context = tf.constant(True, dtype=tf.bool, shape=[self.hparams.context_frames, batch_size])
        self.ground_truth = tf.concat([ground_truth_context, ground_truth_sampling], axis=0)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def zero_state(self, batch_size, dtype):
        init_state = super(VPredCell, self).zero_state(batch_size, dtype)
        init_state['last_images'] = [self.inputs['images'][0]] * self.hparams.last_frames
        if 'pix_distribs' in self.inputs:
            init_state['last_pix_distribs'] = [self.inputs['pix_distribs'][0]] * self.hparams.last_frames
        return init_state

    def _conv_rnn_func(self, inputs, state, filters):
        inputs_shape = inputs.get_shape().as_list()
        input_shape = inputs_shape[1:]
        if self.hparams.norm_layer == 'none':
            normalizer_fn = None
        else:
            normalizer_fn = ops.get_norm_layer(self.hparams.norm_layer)
        if self.hparams.conv_rnn == 'lstm':
            Conv2DRNNCell = BasicConv2DLSTMCell
        elif self.hparams.conv_rnn == 'gru':
            Conv2DRNNCell = Conv2DGRUCell
        else:
            raise NotImplementedError
        if self.hparams.ablation_conv_rnn_norm:
            conv_rnn_cell = Conv2DRNNCell(input_shape, filters, kernel_size=(5, 5),
                                          reuse=tf.get_variable_scope().reuse)
            h, state = conv_rnn_cell(inputs, state)
            outputs = (normalizer_fn(h), state)
        else:
            conv_rnn_cell = Conv2DRNNCell(input_shape, filters, kernel_size=(5, 5),
                                          normalizer_fn=normalizer_fn,
                                          separate_norms=self.hparams.norm_layer == 'layer',
                                          reuse=tf.get_variable_scope().reuse)
            outputs = conv_rnn_cell(inputs, state)
        return outputs

    def call(self, inputs, states):
        norm_layer = ops.get_norm_layer(self.hparams.norm_layer)
        downsample_layer = ops.get_downsample_layer(self.hparams.downsample_layer)
        upsample_layer = ops.get_upsample_layer(self.hparams.upsample_layer)
        image_shape = inputs['images'].get_shape().as_list()
        batch_size, height, width, color_channels = image_shape
        conv_rnn_states = states['conv_rnn_states']

        time = states['time']
        with tf.control_dependencies([tf.assert_equal(time[1:], time[0])]):
            t = tf.to_int32(tf.identity(time[0]))

        image = tf.where(self.ground_truth[t], inputs['images'], states['gen_image'])  # schedule sampling (if any)
        last_images = states['last_images'][1:] + [image]

        if 'pix_distribs' in inputs:
            pix_distrib = tf.where(self.ground_truth[t], inputs['pix_distribs'], states['gen_pix_distrib'])
            last_pix_distribs = states['last_pix_distribs'][1:] + [pix_distrib]
        if 'states' in inputs:
            state = tf.where(self.ground_truth[t], inputs['states'], states['gen_state'])

        state_action = []
        state_action_z = []
        if 'actions' in inputs:
            state_action.append(inputs['actions'])
            state_action_z.append(inputs['actions'])
        if 'states' in inputs:
            state_action.append(state)
            # don't backpropagate the convnet through the state dynamics
            state_action_z.append(tf.stop_gradient(state))

        def concat(tensors, axis):
            if len(tensors) == 0:
                return tf.zeros([batch_size, 0])
            elif len(tensors) == 1:
                return tensors[0]
            else:
                return tf.concat(tensors, axis=axis)
        state_action = concat(state_action, axis=-1)
        state_action_z = concat(state_action_z, axis=-1)
        if 'actions' in inputs:
            gen_input = tile_concat([image, inputs['actions'][:, None, None, :]], axis=-1)
        else:
            gen_input = image

        layers = []
        new_conv_rnn_states = []

        for i, (out_channels, use_conv_rnn) in enumerate(self.encoder_layer_specs):
            with tf.variable_scope('h%d' % i):
                if i == 0:
                    h = tf.concat([image, self.inputs['images'][0]], axis=-1)
                    kernel_size = (5, 5)
                else:
                    h = layers[-1][-1]
                    kernel_size = (3, 3)
                if self.hparams.where_add == 'all' or (self.hparams.where_add == 'input' and i == 0):
                    h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                h = downsample_layer(h, out_channels, kernel_size=kernel_size, strides=(2, 2))
                h = norm_layer(h)
                h = tf.nn.relu(h)
            if use_conv_rnn:
                conv_rnn_state = conv_rnn_states[len(new_conv_rnn_states)]
                with tf.variable_scope('%s_h%d' % (self.hparams.conv_rnn, i)):
                    if self.hparams.where_add == 'all':
                        conv_rnn_h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                    else:
                        conv_rnn_h = h
                    conv_rnn_h, conv_rnn_state = self._conv_rnn_func(conv_rnn_h, conv_rnn_state, out_channels)
                new_conv_rnn_states.append(conv_rnn_state)
            layers.append((h, conv_rnn_h) if use_conv_rnn else (h,))

        num_encoder_layers = len(layers)
        for i, (out_channels, use_conv_rnn) in enumerate(self.decoder_layer_specs):
            with tf.variable_scope('h%d' % len(layers)):
                if i == 0:
                    h = layers[-1][-1]
                else:
                    h = tf.concat([layers[-1][-1], layers[num_encoder_layers - i - 1][-1]], axis=-1)
                if self.hparams.where_add == 'all' or (self.hparams.where_add == 'middle' and i == 0):
                    h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                h = upsample_layer(h, out_channels, kernel_size=(3, 3), strides=(2, 2))
                h = norm_layer(h)
                h = tf.nn.relu(h)
            if use_conv_rnn:
                conv_rnn_state = conv_rnn_states[len(new_conv_rnn_states)]
                with tf.variable_scope('%s_h%d' % (self.hparams.conv_rnn, len(layers))):
                    if self.hparams.where_add == 'all':
                        conv_rnn_h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                    else:
                        conv_rnn_h = h
                    conv_rnn_h, conv_rnn_state = self._conv_rnn_func(conv_rnn_h, conv_rnn_state, out_channels)
                new_conv_rnn_states.append(conv_rnn_state)
            layers.append((h, conv_rnn_h) if use_conv_rnn else (h,))
        assert len(new_conv_rnn_states) == len(conv_rnn_states)

        if self.hparams.last_frames and self.hparams.num_transformed_images:
            if self.hparams.transformation == 'flow':
                with tf.variable_scope('h%d_flow' % len(layers)):
                    h_flow = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                    h_flow = norm_layer(h_flow)
                    h_flow = tf.nn.relu(h_flow)

                with tf.variable_scope('flows'):
                    flows = conv2d(h_flow, 2 * self.hparams.last_frames * self.hparams.num_transformed_images, kernel_size=(3, 3), strides=(1, 1))
                    flows = tf.reshape(flows, [batch_size, height, width, 2, self.hparams.last_frames * self.hparams.num_transformed_images])
            else:
                assert len(self.hparams.c_dna_kernel_size) == 2
                kernel_shape = list(self.hparams.c_dna_kernel_size) + [self.hparams.last_frames * self.hparams.num_transformed_images]
                if self.hparams.transformation == 'dna':
                    with tf.variable_scope('h%d_dna_kernel' % len(layers)):
                        h_dna_kernel = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                        h_dna_kernel = norm_layer(h_dna_kernel)
                        h_dna_kernel = tf.nn.relu(h_dna_kernel)

                    # Using largest hidden state for predicting untied conv kernels.
                    with tf.variable_scope('dna_kernels'):
                        kernels = conv2d(h_dna_kernel, np.prod(kernel_shape), kernel_size=(3, 3), strides=(1, 1))
                        kernels = tf.reshape(kernels, [batch_size, height, width] + kernel_shape)
                        kernels = kernels + identity_kernel(self.hparams.c_dna_kernel_size)[None, None, None, :, :, None]
                    kernel_spatial_axes = [3, 4]
                elif self.hparams.transformation == 'cdna':
                    with tf.variable_scope('cdna_kernels'):
                        smallest_layer = layers[num_encoder_layers - 1][-1]
                        kernels = dense(flatten(smallest_layer), np.prod(kernel_shape))
                        kernels = tf.reshape(kernels, [batch_size] + kernel_shape)
                        kernels = kernels + identity_kernel(self.hparams.c_dna_kernel_size)[None, :, :, None]
                    kernel_spatial_axes = [1, 2]
                else:
                    raise ValueError('Invalid transformation %s' % self.hparams.transformation)

                with tf.name_scope('kernel_normalization'):
                    kernels = tf.nn.relu(kernels - RELU_SHIFT) + RELU_SHIFT
                    kernels /= tf.reduce_sum(kernels, axis=kernel_spatial_axes, keepdims=True)

        if self.hparams.generate_scratch_image:
            with tf.variable_scope('h%d_scratch' % len(layers)):
                h_scratch = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                h_scratch = norm_layer(h_scratch)
                h_scratch = tf.nn.relu(h_scratch)

            # Using largest hidden state for predicting a new image layer.
            # This allows the network to also generate one image from scratch,
            # which is useful when regions of the image become unoccluded.
            with tf.variable_scope('scratch_image'):
                scratch_image = conv2d(h_scratch, color_channels, kernel_size=(3, 3), strides=(1, 1))
                scratch_image = tf.nn.sigmoid(scratch_image)

        with tf.name_scope('transformed_images'):
            transformed_images = []
            if self.hparams.last_frames and self.hparams.num_transformed_images:
                if self.hparams.transformation == 'flow':
                    transformed_images.extend(apply_flows(last_images, flows))
                else:
                    transformed_images.extend(apply_kernels(last_images, kernels, self.hparams.dilation_rate))
            if self.hparams.prev_image_background:
                transformed_images.append(image)
            if self.hparams.first_image_background and not self.hparams.context_images_background:
                transformed_images.append(self.inputs['images'][0])
            if self.hparams.context_images_background:
                transformed_images.extend(tf.unstack(self.inputs['images'][:self.hparams.context_frames]))
            if self.hparams.generate_scratch_image:
                transformed_images.append(scratch_image)

        if 'pix_distribs' in inputs:
            with tf.name_scope('transformed_pix_distribs'):
                transformed_pix_distribs = []
                if self.hparams.last_frames and self.hparams.num_transformed_images:
                    if self.hparams.transformation == 'flow':
                        transformed_pix_distribs.extend(apply_flows(last_pix_distribs, flows))
                    else:
                        transformed_pix_distribs.extend(apply_kernels(last_pix_distribs, kernels, self.hparams.dilation_rate))
                if self.hparams.prev_image_background:
                    transformed_pix_distribs.append(pix_distrib)
                if self.hparams.first_image_background and not self.hparams.context_images_background:
                    transformed_pix_distribs.append(self.inputs['pix_distribs'][0])
                if self.hparams.context_images_background:
                    transformed_pix_distribs.extend(tf.unstack(self.inputs['pix_distribs'][:self.hparams.context_frames]))
                if self.hparams.generate_scratch_image:
                    transformed_pix_distribs.append(pix_distrib)

        with tf.name_scope('masks'):
            if len(transformed_images) > 1:
                with tf.variable_scope('h%d_masks' % len(layers)):
                    h_masks = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                    h_masks = norm_layer(h_masks)
                    h_masks = tf.nn.relu(h_masks)

                with tf.variable_scope('masks'):
                    if self.hparams.dependent_mask:
                        h_masks = tf.concat([h_masks] + transformed_images, axis=-1)
                    masks = conv2d(h_masks, len(transformed_images), kernel_size=(3, 3), strides=(1, 1))
                    masks = tf.nn.softmax(masks)
                    masks = tf.split(masks, len(transformed_images), axis=-1)
            elif len(transformed_images) == 1:
                masks = [tf.ones([batch_size, height, width, 1])]
            else:
                raise ValueError("Either one of the following should be true: "
                                 "last_frames and num_transformed_images, first_image_background, "
                                 "prev_image_background, generate_scratch_image")

        with tf.name_scope('gen_images'):
            assert len(transformed_images) == len(masks)
            gen_image = tf.add_n([transformed_image * mask
                                  for transformed_image, mask in zip(transformed_images, masks)])

        if 'pix_distribs' in inputs:
            with tf.name_scope('gen_pix_distribs'):
                assert len(transformed_pix_distribs) == len(masks)
                gen_pix_distrib = tf.add_n([transformed_pix_distrib * mask
                                            for transformed_pix_distrib, mask in zip(transformed_pix_distribs, masks)])
                if self.hparams.renormalize_pixdistrib:
                    gen_pix_distrib /= tf.reduce_sum(gen_pix_distrib, axis=(1, 2), keepdims=True)

        if 'states' in inputs:
            with tf.name_scope('gen_states'):
                with tf.variable_scope('state_pred'):
                    gen_state = dense(state_action, inputs['states'].shape[-1].value)

        outputs = {'gen_images': gen_image,
                   'gen_inputs': gen_input,
                   'transformed_images': tf.stack(transformed_images, axis=-1),
                   'masks': tf.stack(masks, axis=-1)}
        if 'pix_distribs' in inputs:
            outputs['gen_pix_distribs'] = gen_pix_distrib
            outputs['transformed_pix_distribs'] = tf.stack(transformed_pix_distribs, axis=-1)
        if 'states' in inputs:
            outputs['gen_states'] = gen_state
        if self.hparams.transformation == 'flow':
            outputs['gen_flows'] = flows
            flows_transposed = tf.transpose(flows, [0, 1, 2, 4, 3])
            flows_rgb_transposed = tf_utils.flow_to_rgb(flows_transposed)
            flows_rgb = tf.transpose(flows_rgb_transposed, [0, 1, 2, 4, 3])
            outputs['gen_flows_rgb'] = flows_rgb

        new_states = {'time': time + 1,
                      'gen_image': gen_image,
                      'last_images': last_images,
                      'conv_rnn_states': new_conv_rnn_states}
        if 'pix_distribs' in inputs:
            new_states['gen_pix_distrib'] = gen_pix_distrib
            new_states['last_pix_distribs'] = last_pix_distribs
        if 'states' in inputs:
            new_states['gen_state'] = gen_state
        return outputs, new_states
