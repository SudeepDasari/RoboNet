"""
Boiled down version of SAVP model from https://github.com/alexlee-gk/video_prediction
"""
from robonet.inverse_model.models.base_inverse_model import BaseInverseModel
from robonet.video_prediction.utils import tf_utils
import tensorflow as tf
from collections import OrderedDict
from robonet.video_prediction import losses
from robonet.video_prediction.utils import tf_utils


class DeterministicInverseModel(BaseInverseModel):
    def _model_default_hparams(self):
        return {
            "lr": 0.001,
            "end_lr": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
        }

    def _model_fn(self, model_inputs, model_targets, mode):
        inputs, targets = {}, None
        inputs['start_images'] = model_inputs['images'][:, 0]
        inputs['goal_images'] = model_inputs['images'][:, -1]
        if mode == tf.estimator.ModeKeys.TRAIN:
            inputs['T'] = model_targets['actions'].get_shape().as_list()[1]
            inputs['adim'] = model_targets['actions'].get_shape().as_list()[2]
            inputs['real_actions'] = targets = model_targets['actions']
        else:
            inputs['adim'] = model_inputs['adim']
            inputs['T'] = model_inputs['T']

        # build the graph
        self._model_graph = model_graph = self._graph_class()

        if self._num_gpus <= 1:
            outputs = model_graph.build_graph(mode, inputs, self._hparams, self._graph_scope)
        else:
            # TODO: add multi-gpu support
            raise NotImplementedError
    
        # train
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            lr, optimizer = tf_utils.build_optimizer(self._hparams.lr, self._hparams.beta1, self._hparams.beta2, global_step=global_step)
            loss = losses.l1_loss(targets, outputs['pred_actions'])

            print('computing gradient and train_op')
            g_train_op = optimizer.minimize(loss, global_step=global_step)
            
            est = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=g_train_op)
            scalar_summaries = {}
            if 'ground_truth_sampling_mean' in outputs:
                scalar_summaries['ground_truth_sampling_mean'] = outputs['ground_truth_sampling_mean']
            return est, scalar_summaries, {}
            
        #test
        return outputs['pred_actions']

