import ray
from robonet.inverse_model.models import get_models
import numpy as np
from robonet.video_prediction.utils import tf_utils
import tensorflow as tf
from tensorflow.contrib.training import HParams
import os
import glob
import math
import yaml


class ActionInferenceInterface(object):
    def __init__(self, model_path, test_hparams={}, n_gpus=1, first_gpu=0, sess=None):
        assert n_gpus == 1, "multi gpu evaluation not yet written"
        assert first_gpu == 0, "only starts building at gpu0"
        
        self._test_hparams = self._default_hparams().override_from_dict(test_hparams)
        self._model_path = os.path.expanduser(model_path)

        config_path = self._model_path + '/params.yaml'
        assert os.path.exists(config_path), 'Config path does not exist!'

        with open(config_path) as config:
            params = yaml.load(config, Loader=yaml.SafeLoader)
            self._model_hparams = params['model']
            self._input_hparams = params['dataset']

        # ensure vgg weights are restored correctly (a bit hacky for now)
        self._model_hparams['vgg_path'] = os.path.expanduser(self._test_hparams.vgg_path)

        print('\n\n------------------------------------ LOADED PARAMS ------------------------------------')
        for k, v in self._model_hparams.items():
            print('{} --> {}'.format(k, v))
        for k, v in self._input_hparams.items():
            print('{} --> {}'.format(k, v))
        print('---------------------------------------------------------------------------------------\n\n')
        
        InverseModel = get_models(self._model_hparams.pop('model'))
        self._model = model = InverseModel(self._input_hparams, n_gpus, self._model_hparams['graph_type'], False, self._model_hparams.pop('scope_name'))
        inputs, targets = self._build_input_targets()
        self._pred_act= model.model_fn(inputs, targets, tf.estimator.ModeKeys.PREDICT, self._model_hparams)
        
        self._sess = sess
        self._restored = False
    
    def _default_hparams(self):
        default_dict = {
            "run_batch_size": 1,
            "vgg_path": "~/"                # vgg19.npy should be in vgg_path folder (aka vgg_path = /path/to/folder/containing/weights/)
        }
        return HParams(**default_dict)
    
    def _build_input_targets(self):
        n_context = self._model_hparams.get('context_actions', 0)
        height, width = self._input_hparams['img_size']
        self._images_pl = tf.placeholder(tf.float32, [self._test_hparams.run_batch_size, 2 + n_context, height, width, 3])
        pl_dict = {'adim': self._input_hparams['target_adim'], 'T': self._input_hparams['load_T'] - 1, 'images': self._images_pl}
        
        if n_context:
            self._context_pl = tf.placeholder(tf.float32, [self._test_hparams.run_batch_size, self._model_hparams['context_actions'], 
                                                            self._input_hparams['target_adim']])
            pl_dict['context_actions'] = self._context_pl

        return pl_dict, {}
    
    def predict(self, start_image, goal_image, context_actions=None, context_frames=None):
        assert self._restored
        start_goal_image = np.concatenate((start_image[:, None], goal_image[:, None]), axis=1)
        fd = {self._images_pl: start_goal_image}
        if self._model_hparams.get('context_actions', 0):
            fd[self._images_pl] = np.concatenate((context_frames, start_goal_image), axis=1)
            fd[self._context_pl] = context_actions
        return self._sess.run(self._pred_act, feed_dict=fd)
    
    def __call__(self, start_image, goal_image, context_actions=None, context_frames=None):
        return self.predict(start_image, goal_image, context_actions, context_frames)

    def set_session(self, sess):
        self._sess = sess

    def restore(self):
        if self._sess is None:
            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())

        model_paths = glob.glob('{}/model-*'.format(self._model_path))
        max_model = max([int(m.split('.')[0].split('-')[-1]) for m in model_paths])
        restore_path = os.path.join(self._model_path, 'model-' + str(max_model))
        print('restoring', restore_path)
        
        checkpoints = [restore_path]
        # automatically skip global_step if more than one checkpoint is provided
        skip_global_step = len(checkpoints) > 1
        savers = []
        for checkpoint in checkpoints:
            print("creating restore saver from checkpoint %s" % checkpoint)
            saver, _ = tf_utils.get_checkpoint_restore_saver(checkpoint, skip_global_step=skip_global_step)
            savers.append(saver)
        restore_op = [saver.saver_def.restore_op_name for saver in savers]
        self._sess.run(restore_op)
        self._restored = True
    
    @property
    def horizon(self):
        return self._input_hparams['load_T'] - 1

    @property
    def context_actions(self):
        return self._model_hparams.get('context_actions', 0)
