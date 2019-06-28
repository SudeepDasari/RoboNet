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
from robonet.video_prediction.models.stochastic_generator import create_n_layer_encoder
import pdb

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
        'action_weight': 0.0,
    }


def onestep_encoder_fn(targets, hparams=None):
    image_pairs = tf.concat([targets['images'][:-1], targets['images'][1:]], axis=-1)

    targets = tile_concat([image_pairs, targets['actions'][:-1][:,:, None, None]], axis=-1)

    assert targets.shape.ndims == 5

    batch_shape = targets.shape[:-3].as_list()
    targets = flatten(targets, 0, len(batch_shape) - 1)
    unflatten = lambda x: tf.reshape(x, batch_shape + x.shape.as_list()[1:])
    outputs = create_n_layer_encoder(targets, stochastic=hparams.stochastic)
    return nest.map_structure(unflatten, outputs)


def split_model_inference(inputs, targets, params):
    """
    we use separate trajectories for the encoder than from the ones used for prediction training
    :param inputs: dict with tensors in *time-major*
    :param targets:dict with tensors in *time-major*
    :return:
    """
    def split(inputs, bs, sbs):
        first_half = {}
        second_half = {}
        for key, value in inputs.items():
            first_half[key] = []
            second_half[key] = []
            for i in range(bs // sbs):
                first_half[key].append(value[:, sbs * i:sbs * i + sbs // 2])
                second_half[key].append(value[:, sbs * i + sbs // 2:sbs * (i + 1)])
            first_half[key] = tf.concat(first_half[key], 1)
            second_half[key] = tf.concat(second_half[key], 1)
        return first_half, second_half

    sbs = params.sub_batch_size
    bs = params.batch_size
    inputs_train, inputs_inference = split(inputs, bs, sbs)
    targets_train, targets_inference = split(targets, bs, sbs)

    return {'train':inputs_train, 'inference':inputs_inference}, \
           {'train':targets_train, 'inference':targets_inference}


def average_and_repeat(enc, params, tlen):
    """
    :param enc:  time, batch, z_dim
    :param params:
    :param tlen: length of horizon
    :return: e in time-major
    """
    enc = tf.reduce_mean(enc, axis=0)   # average over time dimension
    hsbs = params.sub_batch_size // 2
    bs = params.batch_size
    e = []
    for i in range(bs // params.sub_batch_size):
        averaged = tf.reduce_mean(enc[i*hsbs: (i+1)*hsbs], axis=0)  # average over sub-batch dimension
        averaged = tf.tile(averaged[None], [hsbs, 1])  # tile across sub-batch
        e.append(averaged)
    e = tf.concat(e, axis=0)
    e = tf.tile(e[None], [tlen, 1, 1])
    return e


def deterministic_embedding_generator(num_gpus, graph_type, tpu_mode, inputs, targets, mode, params):
    # get graph class and setup default hyper-parameters
    logger = logging.getLogger(__name__)
    graph_class = get_graph_class(graph_type)
    default_hparams = dict(itertools.chain(graph_class.default_hparams().items(), loss_default_hparams(graph_class).items()))
    hparams = HParams(**default_hparams).override_from_dict(params)

    # prep inputs here, convert to time-major
    inputs['actions'], inputs['images'], inputs['states'] = tf.transpose(inputs['actions'], [1, 0, 2]), tf.transpose(inputs['images'], [1, 0, 2, 3, 4]), tf.transpose(inputs['states'], [1,0,2])
    targets['images'], targets['states'] = tf.transpose(targets['images'], [1, 0, 2, 3, 4]), tf.transpose(targets['states'], [1,0,2])

    tlen = inputs['images'].get_shape().as_list()[0]
    inputs, targets = split_model_inference(inputs, targets, hparams)

    if hparams.encoder == 'one_step':
        outputs_enc = onestep_encoder_fn(inputs['inference'], hparams)
        hparams.e_dim = outputs_enc.get_shape().as_list()[2]
        outputs_enc = average_and_repeat(outputs_enc, hparams, tlen)
    else:
        raise NotImplementedError

    target_images = targets['train']['images'][hparams.context_frames:]

    # build the graph
    model_graph = graph_class()
    if num_gpus == 1:
        outputs = model_graph.build_graph(inputs['train'], hparams, outputs_enc=outputs_enc)
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

        scalar_summaries, tensor_summaries = {}, {'pred_frames': pred_frames, 'pred_targets':target_images, 'inference_images':inputs['inference']['images']}
        
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

        loss = sum(loss * weight for loss, weight in gen_losses.values())
        g_gradvars = optimizer.compute_gradients(loss, var_list=model_graph.vars)
        g_train_op = optimizer.apply_gradients(g_gradvars, global_step=global_step)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=g_train_op, predictions=pred_frames), scalar_summaries, tensor_summaries
    # if test build the predictor
    raise NotImplementedError