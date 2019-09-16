"""
Boiled down version of SAVP model from https://github.com/alexlee-gk/video_prediction
"""
from robonet.inverse_model.models.base_inverse_model import BaseInverseModel
from robonet.video_prediction.utils import tf_utils
import tensorflow as tf
from collections import OrderedDict
from robonet.video_prediction import losses
from robonet.video_prediction.utils import tf_utils


class DiscretizedInverseModel(BaseInverseModel):
    def _model_default_hparams(self):
        return {
            "lr": 0.001,
            "end_lr": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
            "pivots": [[-0.04483253140755173, -0.02947711320550581, -0.018373884708696702, -0.008892051974322548, -4.59881939272745e-05, 0.008815899693566963, 0.018292582474913204, 0.02938255920278165, 0.04470332342338521],
                        [-0.044674549010427486, -0.029352782231283018, -0.018263887904468375, -0.008836470630237072, 7.81874877900302e-06, 0.00884825636063618, 0.01830693463003378, 0.029377939442953, 0.04473508111072804],
                        [-0.10348141529525286, -0.06793363038544242, -0.042405628783200776, -0.02067683018449292, -0.0003540274691179853, 0.019988218195319766, 0.04168513725690283, 0.06726936589279635, 0.10260515613003221],
                        [-0.22409500837108018, -0.1470685835529137, -0.09166876049855337, -0.04419968109806307, 5.580875190224738e-05, 0.044414223320168145, 0.09168509202611021, 0.1469321233733917, 0.2237400683241968]]
        }

    def _model_fn(self, model_inputs, model_targets, mode):
        inputs, targets = {}, None
        inputs['start_images'] = model_inputs['images'][:, 0]
        inputs['goal_images'] = model_inputs['images'][:, -1]
        if mode == tf.estimator.ModeKeys.TRAIN:
            B = model_targets['actions'].get_shape().as_list()[0]
            inputs['T'] = model_targets['actions'].get_shape().as_list()[1]
            input_adim = model_targets['actions'].get_shape().as_list()[2]

            assert input_adim == 4, "only supports [x,y,z,theta] action space for now!"
            assert len(self._hparams.pivots) == input_adim, "bad discretization pivots array!"
            binned_actions = []
            for a in range(input_adim):
                binned_action = tf.zeros((B, inputs['T']), dtype=tf.int32)
                for p in range(len(self._hparams.pivots)):
                    binned_action = tf.where_v2(model_targets['actions'][:, :, a] > p, binned_action + 1, binned_action)
                binned_actions.append(binned_action)
            xy_act = binned_actions[0] + (len(self._hparams.pivots[0]) + 1) + binned_actions[1]
            n_xy = (len(self._hparams.pivots[0]) + 1) * (len(self._hparams.pivots[1]) + 1)
            z_act, n_z = binned_actions[2], len(self._hparams.pivots[2]) + 1
            theta_act, n_theta = binned_actions[3], len(self._hparams.pivots[3]) + 1
            one_hot_actions = [tf.one_hot(tensor, n_dim) for tensor, n_dim in zip((xy_act, z_act, theta_act), (n_xy, n_z, n_theta))]
            inputs['real_actions'] = tf.concat(one_hot_actions, -1)
        else:
            assert model_inputs['adim'] == 4, "only supports [x,y,z,theta] action space for now!"
            inputs['T'] = model_inputs['T']
        inputs['adim'] = (len(self._hparams.pivots[0]) + 1) * (len(self._hparams.pivots[1]) + 1) + sum([len(arr) + 1 for arr in self._hparams.pivots[2:]])

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
            pred_xy = outputs['pred_actions'][:, :, :n_xy]
            pred_z = outputs['pred_actions'][:, :, n_xy:n_z + n_xy]
            pred_theta = outputs['pred_actions'][:, :, n_z + n_xy:]
            pred_one_hots = [pred_xy, pred_z, pred_theta]

            losses = [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(real, pred)) for real, pred in zip(one_hot_actions, pred_one_hots)]
            loss = sum(losses)

            print('computing gradient and train_op')
            g_train_op = optimizer.minimize(loss, global_step=global_step)
            
            est = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=g_train_op)
            scalar_summaries = {}
            if 'ground_truth_sampling_mean' in outputs:
                scalar_summaries['ground_truth_sampling_mean'] = outputs['ground_truth_sampling_mean']
            
            for k, loss in zip(['xy_loss', 'z_loss', 'theta_loss'], losses):
                scalar_summaries[k] = loss
            return est, scalar_summaries, {}
            
        #test
        return outputs['pred_actions']

