"""
Boiled down version of SAVP model from https://github.com/alexlee-gk/video_prediction
"""
from robonet.inverse_model.models.base_inverse_model import BaseInverseModel
from robonet.video_prediction.utils import tf_utils
import tensorflow as tf
from collections import OrderedDict
from robonet.video_prediction import losses
from robonet.video_prediction.utils import tf_utils


class VAEInverse(BaseInverseModel):
    def _model_default_hparams(self):
        return {
            "lr": 0.001,
            "end_lr": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
            'l1_weight': 1.0,
            'kl_weight': 1.0,
        }

    def _model_fn(self, model_inputs, model_targets, mode):
        inputs, targets = {}, None
        inputs['start_images'] = model_targets['images'][:, 0]
        inputs['goal_images'] = model_targets['images'][:, -1]
        inputs['T'] = model_inputs['actions'].get_shape().as_list()[1]
        inputs['adim'] = model_inputs['actions'].get_shape().as_list()[2]
        targets = model_inputs['actions']

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
            
            kl_loss = losses.kl_loss(outputs['kl_mu'], 2 * outputs['kl_logsigma'])
            l1_loss = losses.l1_loss(targets, outputs['pred_actions'])
            loss = l1_loss * self._hparams.l1_weight + kl_loss * self._hparams.kl_weight

            print('computing gradient and train_op')
            g_gradvars = optimizer.compute_gradients(loss, var_list=model_graph.vars)
            g_train_op = optimizer.apply_gradients(g_gradvars, global_step=global_step)
            
            est = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=g_train_op)
            return est, { 'l1_loss': l1_loss}, {}
            
        #test
        raise NotImplementedError

