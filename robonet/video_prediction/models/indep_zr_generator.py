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
        'zat_kl_weight': 0.001
    }


def vpred_generator(num_gpus, graph_type, inputs, targets, mode, params):
    # get graph class and setup default hyper-parameters
    logger = logging.getLogger(__name__)
    graph_class = get_graph_class(graph_type)
    default_hparams = dict(itertools.chain(graph_class.default_hparams().items(), loss_default_hparams(graph_class).items()))
    hparams = HParams(**default_hparams).override_from_dict(params)
    
    # prep inputs here
    inputs['actions'], inputs['images'] = tf.transpose(inputs['actions'], [1, 0, 2]), tf.transpose(inputs['images'], [1, 0, 2, 3, 4])
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
        outputs = model_graph.build_graph(inputs, hparams)
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
        
        if hparams.l1_weight:
            gen_l1_loss = losses.l1_loss(gen_images, target_images)
            gen_losses["gen_l1_loss"] = (gen_l1_loss, hparams.l1_weight)
        if hparams.l2_weight:
            gen_l2_loss = losses.l2_loss(gen_images, target_images)
            gen_losses["gen_l2_loss"] = (gen_l2_loss, hparams.l2_weight)
        if (hparams.l1_weight or hparams.l2_weight) and hparams.num_scales > 1:
            for i in range(1, hparams.num_scales):
                scale_factor = 2 ** i
                gen_images_scale = tf_utils.with_flat_batch(pool2d)(gen_images, scale_factor, scale_factor, pool_mode='avg')
                target_images_scale = tf_utils.with_flat_batch(pool2d)(target_images, scale_factor, scale_factor, pool_mode='avg')
                if hparams.l1_weight:
                    gen_l1_scale_loss = losses.l1_loss(gen_images_scale, target_images_scale)
                    gen_losses["gen_l1_scale%d_loss" % i] = (gen_l1_scale_loss, hparams.l1_weight)
                if hparams.l2_weight:
                    gen_l2_scale_loss = losses.l2_loss(gen_images_scale, target_images_scale)
                    gen_losses["gen_l2_scale%d_loss" % i] = (gen_l2_scale_loss, hparams.l2_weight)
        if hparams.vgg_cdist_weight:
            gen_vgg_cdist_loss = metrics.vgg_cosine_distance(gen_images, target_images)
            gen_losses['gen_vgg_cdist_loss'] = (gen_vgg_cdist_loss, hparams.vgg_cdist_weight)
        if hparams.state_weight:
            gen_states = outputs.get('gen_states_enc', outputs['gen_states'])
            target_states = targets['states']
            gen_state_loss = losses.l2_loss(gen_states, target_states)
            gen_losses["gen_state_loss"] = (gen_state_loss, hparams.state_weight)
        if hparams.tv_weight:
            gen_flows = outputs.get('gen_flows_enc', outputs['gen_flows'])
            flow_diff1 = gen_flows[..., 1:, :, :, :] - gen_flows[..., :-1, :, :, :]
            flow_diff2 = gen_flows[..., :, 1:, :, :] - gen_flows[..., :, :-1, :, :]
            # sum over the multiple transformations but take the mean for the other dimensions
            gen_tv_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(flow_diff1), axis=(-2, -1))) + \
                          tf.reduce_mean(tf.reduce_sum(tf.abs(flow_diff2), axis=(-2, -1)))
            gen_losses['gen_tv_loss'] = (gen_tv_loss, hparams.tv_weight)
        if hparams.action_weight:
            gen_actions = outputs['gen_actions']
            target_actions = inputs['actions'][hparams.context_frames-1:]
            gen_action_loss = losses.l2_loss(gen_actions, target_actions)
            gen_losses["gen_action_loss"] = (gen_action_loss, hparams.action_weight)
        if hparams.zat_kl_weight:   # TODO: add annealing
            gen_zat_kl_loss = losses.kl_loss(outputs['zat_mu'], outputs['zat_log_sigma_sq'])
            gen_losses["gen_zat_kl_loss"] = (gen_zat_kl_loss, hparams.zat_kl_weight)  # possibly annealed kl_weight

        loss = sum(loss * weight for loss, weight in gen_losses.values())
        g_gradvars = optimizer.compute_gradients(loss, var_list=model_graph.vars)
        g_train_op = optimizer.apply_gradients(g_gradvars, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=g_train_op, predictions=pred_frames)
    # if test build the predictor
    raise NotImplementedError