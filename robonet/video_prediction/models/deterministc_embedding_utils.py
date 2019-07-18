"""
TODO: use self._data_hparams instead of hacking batch_size/sub_batch_size into the model_hparams

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
from robonet.video_prediction.layers.encoder_layers import create_n_layer_encoder


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

