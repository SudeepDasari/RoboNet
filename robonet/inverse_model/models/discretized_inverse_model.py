"""
Boiled down version of SAVP model from https://github.com/alexlee-gk/video_prediction
"""
from robonet.inverse_model.models.base_inverse_model import BaseInverseModel
from robonet.video_prediction.utils import tf_utils
import tensorflow as tf
from collections import OrderedDict
from robonet.video_prediction import losses
from robonet.video_prediction.utils import tf_utils


def _binarize(actions, pivots):
    n_xy = (len(pivots[0]) + 1) * (len(pivots[1]) + 1)
    n_z =  len(pivots[2]) + 1
    n_theta = len(pivots[3]) + 1

    B = actions.get_shape().as_list()[0]
    input_adim = actions.get_shape().as_list()[2]
    T = actions.get_shape().as_list()[1]

    assert input_adim == 4, "only supports [x,y,z,theta] action space for now!"
    assert len(pivots) == input_adim, "bad discretization pivots array!"
    binned_actions = []
    for a in range(input_adim):
        binned_action = tf.zeros((B, T), dtype=tf.int32)
        for p in range(len(pivots[a])):
            pivot = pivots[a][p]
            binned_action = tf.where_v2(actions[:, :, a] > pivot, binned_action + 1, binned_action)
        binned_actions.append(binned_action)
    
    xy_act = binned_actions[0] + (len(pivots[0]) + 1) * binned_actions[1]
    z_act, theta_act = binned_actions[2], binned_actions[3]
    one_hot_actions = [tf.one_hot(tensor, n_dim) for tensor, n_dim in zip((xy_act, z_act, theta_act), (n_xy, n_z, n_theta))]
    return one_hot_actions


class DiscretizedInverseModel(BaseInverseModel):
    def _model_default_hparams(self):
        return {
            "context_actions": 0,
            "lr": 0.001,
            "end_lr": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
            "pivots": [[-0.04483253140755173, -0.02947711320550581, -0.018373884708696702, -0.008892051974322548, -4.59881939272745e-05, 0.008815899693566963, 0.018292582474913204, 0.02938255920278165, 0.04470332342338521],
                        [-0.044674549010427486, -0.029352782231283018, -0.018263887904468375, -0.008836470630237072, 7.81874877900302e-06, 0.00884825636063618, 0.01830693463003378, 0.029377939442953, 0.04473508111072804],
                        [-0.10348141529525286, -0.06793363038544242, -0.042405628783200776, -0.02067683018449292, -0.0003540274691179853, 0.019988218195319766, 0.04168513725690283, 0.06726936589279635, 0.10260515613003221],
                        [-0.22409500837108018, -0.1470685835529137, -0.09166876049855337, -0.04419968109806307, 5.580875190224738e-05, 0.044414223320168145, 0.09168509202611021, 0.1469321233733917, 0.2237400683241968]],
            "means": [[-0.05844043352506317, -0.0365598888753108, -0.02371854361080623, -0.01355633452537272, -0.004447061217304071, 0.004359558603466982, 0.01346781084244209, 0.02363783086130393, 0.036456939880113295, 0.05834560772861528],
                      [-0.05831025927528526, -0.03643373528153938, -0.023609139710274608, -0.013465667182953755, -0.004399357688117235, 0.004405043570967748, 0.013491056349448851, 0.023632353969085647, 0.03646405448080863, 0.0583175660974888],
                      [-0.14210753817324154, -0.08433897448430323, -0.054693763651882464, -0.03133158710778195, -0.010471111756616646, 0.00976338713468559, 0.030621494148596932, 0.05401384675615853, 0.08356642563278535, 0.1406814351222195],
                      [-0.30673709675244115, -0.1826234528754964, -0.11831810064105407, -0.06747048133665953, -0.02199376800432712, 0.02209506703978301, 0.06762712392804507, 0.11832652238765545, 0.18242774553595653, 0.30635348910031857]]
        }

    def _model_fn(self, model_inputs, model_targets, mode):
        inputs = {}
        if self._hparams.context_actions:
            inputs['context_frames'] = model_inputs['images'][:, :self._hparams.context_actions]
        inputs['start_images'] = model_inputs['images'][:, self._hparams.context_actions]
        inputs['goal_images'] = model_inputs['images'][:, -1]

        n_xy = (len(self._hparams.pivots[0]) + 1) * (len(self._hparams.pivots[1]) + 1)
        n_z =  len(self._hparams.pivots[2]) + 1
        n_theta = len(self._hparams.pivots[3]) + 1

        if mode == tf.estimator.ModeKeys.TRAIN:
            one_hot_actions = _binarize(model_targets['actions'], self._hparams.pivots)
            if self._hparams.context_actions:
                inputs['context_actions'] = tf.concat([x[:, :self._hparams.context_actions] for x in one_hot_actions], -1)
            real_pred_actions = [x[:, self._hparams.context_actions:] for x in one_hot_actions]
            inputs['real_actions'] = tf.concat(real_pred_actions, -1)
            inputs['T'] = model_targets['actions'].get_shape().as_list()[1] - self._hparams.context_actions
        else:
            assert model_inputs['adim'] == 4, "only supports [x,y,z,theta] action space for now!"
            inputs['T'] = model_inputs['T'] - self._hparams.context_actions
            if self._hparams.context_actions:
                one_hot_actions = _binarize(model_inputs['context_actions'], self._hparams.pivots)
                inputs['context_actions'] = tf.concat([x[:, :self._hparams.context_actions] for x in one_hot_actions], -1)

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

            losses = [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(real, pred)) for real, pred in zip(real_pred_actions, pred_one_hots)]
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
        means = tf.convert_to_tensor(self._hparams.means)
        pred_xy = outputs['pred_actions'][:, :, :n_xy]
        pred_z = outputs['pred_actions'][:, :, n_xy:n_z + n_xy]
        pred_theta = outputs['pred_actions'][:, :, n_z + n_xy:]

        pred_xy = tf.reshape(tf.random.categorical(tf.reshape(pred_xy, (-1, n_xy)), 1, dtype=tf.int32), (-1, inputs['T']))
        pred_x, pred_y = tf.mod(pred_xy, len(self._hparams.pivots[0]) + 1), tf.floordiv(pred_xy, len(self._hparams.pivots[0]) + 1)
        pred_z = tf.reshape(tf.random.categorical(tf.reshape(pred_z, (-1, n_z)), 1, dtype=tf.int32), (-1, inputs['T']))
        pred_theta = tf.reshape(tf.random.categorical(tf.reshape(pred_theta, (-1, n_theta)), 1, dtype=tf.int32), (-1, inputs['T']))

        outputs['pred_actions'] = tf.concat([tf.gather(means[i], indices)[:, :, None] for i, indices in 
                                            enumerate([pred_x, pred_y, pred_z, pred_theta])], axis=-1)
        return outputs['pred_actions']
