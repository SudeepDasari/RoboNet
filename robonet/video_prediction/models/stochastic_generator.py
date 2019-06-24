"""
Boiled down version of SAVP model from https://github.com/alexlee-gk/video_prediction
"""
import itertools
from robonet.video_prediction.utils import tf_utils
import tensorflow as tf
from robonet.video_prediction.models import get_graph_class
from tensorflow.contrib.training import HParams
import logging
from collections import OrderedDict
from robonet.video_prediction import losses
from robonet.video_prediction.ops import lrelu, dense, pad2d, conv2d, conv_pool2d, flatten, tile_concat, pool2d, get_norm_layer
from tensorflow.python.util import nest


def loss_default_hparams(graph_class):
    return {
        "lr": 0.001,
        "end_lr": 0.0,
        "decay_steps": (200000, 300000),
        "max_steps": 300000,
        "beta1": 0.9,
        "beta2": 0.999,
        'l1_weight': 1.0,
        'l2_weight': 0.0,
        'num_scales': 1,
        'vgg_cdist_weight': 0.0,
        'state_weight': 0.0,
        'tv_weight': 0.001,
        'action_weight': 1.0,
        'zat_kl_weight': 0.001,
        'zr_kl_weight': 0.0   # if nonzero, model infers zr from image pairs and actions (if we want to generalize to new domain at test time)
                                # otherwise, model maintains independent zrs (no generalization to new domain at test time)
    }


def create_n_layer_encoder(inputs,
                           nz=8,
                           nef=64,
                           n_layers=3,
                           norm_layer='instance',
                           include_top=True):
    norm_layer = get_norm_layer(norm_layer)
    layers = []
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

    with tf.variable_scope("layer_1"):
        convolved = conv2d(tf.pad(inputs, paddings), nef, kernel_size=4, strides=2, padding='VALID')
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    for i in range(1, n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = nef * min(2**i, 4)
            convolved = conv2d(tf.pad(layers[-1], paddings), out_channels, kernel_size=4, strides=2, padding='VALID')
            normalized = norm_layer(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    pooled = pool2d(rectified, rectified.shape[1:3].as_list(), padding='VALID', pool_mode='avg')
    squeezed = tf.squeeze(pooled, [1, 2])

    if include_top:
        with tf.variable_scope('z_mu'):
            z_mu = dense(squeezed, nz)
        with tf.variable_scope('z_log_sigma_sq'):
            z_log_sigma_sq = dense(squeezed, nz)
            z_log_sigma_sq = tf.clip_by_value(z_log_sigma_sq, -10, 10)
        outputs = {'enc_zs_mu': z_mu, 'enc_zs_log_sigma_sq': z_log_sigma_sq}
    else:
        outputs = squeezed
    return outputs


def create_encoder(inputs, nz):
    assert inputs.shape.ndims == 5
    batch_shape = inputs.shape[:-3].as_list()
    inputs = flatten(inputs, 0, len(batch_shape) - 1)
    unflatten = lambda x: tf.reshape(x, batch_shape + x.shape.as_list()[1:])
    outputs = create_n_layer_encoder(inputs, nz=nz)
    outputs = nest.map_structure(unflatten, outputs)
    return outputs


def encoder_fn(inputs, targets, hparams=None):
    image_pairs = tf.concat([inputs['images'], targets['images']], axis=-1)
    if 'actions' in inputs:
        image_pairs = tile_concat([image_pairs,
                                   tf.expand_dims(tf.expand_dims(inputs['actions'], axis=-2), axis=-2)], axis=-1)
    outputs = create_encoder(image_pairs, nz=hparams.zr_dim)
    return outputs


def vpred_generator(num_gpus, graph_type, tpu_mode, inputs, targets, mode, params):
    # get graph class and setup default hyper-parameters
    logger = logging.getLogger(__name__)
    graph_class = get_graph_class(graph_type)
    default_hparams = dict(itertools.chain(graph_class.default_hparams().items(), loss_default_hparams(graph_class).items()))
    hparams = HParams(**default_hparams).override_from_dict(params)

    # prep inputs here
    inputs['actions'], inputs['images'] = tf.transpose(inputs['actions'], [1, 0, 2]), tf.transpose(inputs['images'], [1, 0, 2, 3, 4])
    if hparams.zr_kl_weight:
        outputs_enc = encoder_fn(inputs, {'images': tf.transpose(targets['images'][:, 1:], [1, 0, 2, 3, 4])}, hparams)
    else:
        outputs_enc = None
    targets['images'] = tf.transpose(targets['images'][:, hparams.context_frames:], [1, 0, 2, 3, 4])
    if hparams.use_states:
        inputs['states'] = tf.transpose(inputs['states'][:, hparams.context_frames:], [1, 0, 2])
        if hparams.state_weight:
            targets['states'] = tf.transpose(targets['state'], [1, 0, 2])
        else:
            logger.warning('states supplied but state_weight=0 so no loss will be computed on predicted states')
    elif hparams.state_weight > 0:
        raise ValueError("states not supplied but state_weight > 0")

    # build the graph
    model_graph = graph_class()
    if num_gpus == 1:
        outputs = model_graph.build_graph(inputs, hparams, outputs_enc=outputs_enc)
    else:
        # TODO: add multi-gpu evaluation support
        raise NotImplementedError
    pred_frames = tf.transpose(outputs["gen_images"], [1,0,2,3,4])

    # if train build the loss function (don't support multi-gpu training)
    if mode == tf.estimator.ModeKeys.TRAIN:
        assert num_gpus == 1, "only single gpu training supported at the moment"
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf_utils.build_optimizer(hparams.lr, hparams.beta1, hparams.beta2, 
                                    decay_steps=hparams.decay_steps, 
                                    end_lr=hparams.end_lr,
                                    global_step=global_step)[1]

        gen_losses = OrderedDict()
        if not (hparams.l1_weight or hparams.l2_weight or hparams.vgg_cdist_weight):
            logger.error('no image loss is being created!')
            raise ValueError
        
        gen_images = outputs.get('gen_images_enc', outputs['gen_images'])
        target_images = targets['images']

        scalar_summaries, tensor_summaries = {}, {'pred_frames': pred_frames}
        
        if hparams.l1_weight:
            gen_l1_loss = losses.l1_loss(gen_images, target_images)
            gen_losses["gen_l1_loss"] = (gen_l1_loss, hparams.l1_weight)
            scalar_summaries['l1_loss'] = gen_l1_loss

        if hparams.l2_weight:
            gen_l2_loss = losses.l2_loss(gen_images, target_images)
            gen_losses["gen_l2_loss"] = (gen_l2_loss, hparams.l2_weight)
            scalar_summaries['l2_loss'] = gen_l2_loss

        if (hparams.l1_weight or hparams.l2_weight) and hparams.num_scales > 1:
            for i in range(1, hparams.num_scales):
                scale_factor = 2 ** i
                gen_images_scale = tf_utils.with_flat_batch(pool2d)(gen_images, scale_factor, scale_factor, pool_mode='avg')
                target_images_scale = tf_utils.with_flat_batch(pool2d)(target_images, scale_factor, scale_factor, pool_mode='avg')
                if hparams.l1_weight:
                    gen_l1_scale_loss = losses.l1_loss(gen_images_scale, target_images_scale)
                    gen_losses["gen_l1_scale%d_loss" % i] = (gen_l1_scale_loss, hparams.l1_weight)
                    scalar_summaries['l1_loss_scale{}'.format(i)] = gen_l1_scale_loss

                if hparams.l2_weight:
                    gen_l2_scale_loss = losses.l2_loss(gen_images_scale, target_images_scale)
                    gen_losses["gen_l2_scale%d_loss" % i] = (gen_l2_scale_loss, hparams.l2_weight)
                    scalar_summaries['l2_loss_scale{}'.format(i)] = gen_l2_scale_loss

        if hparams.vgg_cdist_weight:
            gen_vgg_cdist_loss = metrics.vgg_cosine_distance(gen_images, target_images)
            gen_losses['gen_vgg_cdist_loss'] = (gen_vgg_cdist_loss, hparams.vgg_cdist_weight)
            scalar_summaries['vgg_cdist_loss'] = gen_vgg_cdist_loss

        if hparams.state_weight:
            gen_states = outputs.get('gen_states_enc', outputs['gen_states'])
            target_states = targets['states']
            gen_state_loss = losses.l2_loss(gen_states, target_states)
            gen_losses["gen_state_loss"] = (gen_state_loss, hparams.state_weight)
            scalar_summaries['state_loss'] = gen_state_loss

        if hparams.tv_weight:
            gen_flows = outputs.get('gen_flows_enc', outputs['gen_flows'])
            flow_diff1 = gen_flows[..., 1:, :, :, :] - gen_flows[..., :-1, :, :, :]
            flow_diff2 = gen_flows[..., :, 1:, :, :] - gen_flows[..., :, :-1, :, :]
            # sum over the multiple transformations but take the mean for the other dimensions
            gen_tv_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(flow_diff1), axis=(-2, -1))) + \
                          tf.reduce_mean(tf.reduce_sum(tf.abs(flow_diff2), axis=(-2, -1)))
            gen_losses['gen_tv_loss'] = (gen_tv_loss, hparams.tv_weight)
            scalar_summaries['tv_loss'] = gen_tv_loss

        if hparams.action_weight:
            gen_actions = outputs['gen_actions']
            target_actions = inputs['actions'][hparams.context_frames-1:]
            gen_action_loss = losses.l2_loss(gen_actions, target_actions)
            gen_losses["gen_action_loss"] = (gen_action_loss, hparams.action_weight)
            scalar_summaries['action_loss'] = gen_action_loss

        if hparams.zat_kl_weight:   # TODO: add annealing
            gen_zat_kl_loss = losses.kl_loss(outputs['zat_mu'], outputs['zat_log_sigma_sq'])
            gen_losses["zat_kl_loss"] = (gen_zat_kl_loss, hparams.zat_kl_weight)  # possibly annealed kl_weight
            scalar_summaries['zat_kl_loss'] = gen_zat_kl_loss

        if hparams.zr_kl_weight:    # TODO: add annealing
            gen_zr_kl_loss = losses.kl_loss(outputs['zr_mu'], outputs['zr_log_sigma_sq'])
            gen_losses["gen_zr_kl_loss"] = (gen_zr_kl_loss, hparams.zr_kl_weight)  # possibly annealed kl_weight
            scalar_summaries['gen_zr_kl_loss'] = gen_zr_kl_loss

        loss = sum(loss * weight for loss, weight in gen_losses.values())
        g_gradvars = optimizer.compute_gradients(loss, var_list=model_graph.vars)
        g_train_op = optimizer.apply_gradients(g_gradvars, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=g_train_op, predictions=pred_frames), scalar_summaries, tensor_summaries
    # if test build the predictor
    raise NotImplementedError