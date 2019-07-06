from robonet.video_prediction.layers.dnaflow_rnn_cell import VPredCell
import numpy as np
from robonet.video_prediction.layers.dnaflow_rnn_cell import apply_flows, apply_cdna_kernels, apply_dna_kernels, apply_kernels, identity_kernel, RELU_SHIFT
import tensorflow as tf
from robonet.video_prediction import ops, flow_ops
from robonet.video_prediction.ops import dense, conv2d, flatten, tile_concat
from robonet.video_prediction.rnn_ops import BasicConv2DLSTMCell, Conv2DGRUCell
from robonet.video_prediction.utils import tf_utils

import pdb

class DetVPredCell(VPredCell):
    def __init__(self, mode, inputs, hparams, reuse=None):
        super(DetVPredCell, self).__init__(mode, inputs, hparams, reuse)

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

        state_action_e = []
        if 'actions' in inputs:
            state_action_e.append(inputs['actions'])
        if 'states' in inputs:
            state_action_e.append(state)
        if 'e' in inputs:
            state_action_e.append(inputs['e'])

        def concat(tensors, axis):
            if len(tensors) == 0:
                return tf.zeros([batch_size, 0])
            elif len(tensors) == 1:
                return tensors[0]
            else:
                return tf.concat(tensors, axis=axis)

        state_action_e = concat(state_action_e, axis=-1)

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
                    h = tile_concat([h, state_action_e[:, None, None, :]], axis=-1)
                h = downsample_layer(h, out_channels, kernel_size=kernel_size, strides=(2, 2))
                h = norm_layer(h)
                h = tf.nn.relu(h)
            if use_conv_rnn:
                conv_rnn_state = conv_rnn_states[len(new_conv_rnn_states)]
                with tf.variable_scope('%s_h%d' % (self.hparams.conv_rnn, i)):
                    if self.hparams.where_add == 'all':
                        conv_rnn_h = tile_concat([h, state_action_e[:, None, None, :]], axis=-1)
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
                    h = tile_concat([h, state_action_e[:, None, None, :]], axis=-1)
                h = upsample_layer(h, out_channels, kernel_size=(3, 3), strides=(2, 2))
                h = norm_layer(h)
                h = tf.nn.relu(h)
            if use_conv_rnn:
                conv_rnn_state = conv_rnn_states[len(new_conv_rnn_states)]
                with tf.variable_scope('%s_h%d' % (self.hparams.conv_rnn, len(layers))):
                    if self.hparams.where_add == 'all':
                        conv_rnn_h = tile_concat([h, state_action_e[:, None, None, :]], axis=-1)
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
                    flows = conv2d(h_flow, 2 * self.hparams.last_frames * self.hparams.num_transformed_images,
                                   kernel_size=(3, 3), strides=(1, 1))
                    flows = tf.reshape(flows, [batch_size, height, width, 2,
                                               self.hparams.last_frames * self.hparams.num_transformed_images])
            else:
                assert len(self.hparams.c_dna_kernel_size) == 2
                kernel_shape = list(self.hparams.c_dna_kernel_size) + [
                    self.hparams.last_frames * self.hparams.num_transformed_images]
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
                        transformed_pix_distribs.extend(
                            apply_kernels(last_pix_distribs, kernels, self.hparams.dilation_rate))
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
                    gen_state = dense(state_action_e, inputs['states'].shape[-1].value)

        outputs = {'gen_images': gen_image,
                   'gen_inputs': gen_input,
                   'transformed_images': tf.stack(transformed_images, axis=-1),
                   'masks': tf.stack(masks, axis=-1)}
        if 'pix_distribs' in inputs:
            outputs['gen_pix_distribs'] = gen_pix_distrib
            outputs['transformed_pix_distribs'] = tf.stack(transformed_pix_distribs, axis=-1)
        if 'states' in inputs:
            outputs['gen_states'] = gen_state

        if 'e' in inputs:
            outputs['e'] = inputs['e']
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
