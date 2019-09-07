import ray
from robonet.video_prediction.models import get_model
import numpy as np
import yaml
from robonet.video_prediction.utils import tf_utils
import tensorflow as tf
from tensorflow.contrib.training import HParams
import os
import math
import glob


class VPredEvaluation(object):
    def __init__(self, model_path, test_hparams={}, n_gpus=1, first_gpu=0, sess=None):
        assert n_gpus == 1, "multi gpu evaluation not yet written"
        assert first_gpu == 0, "only starts building at gpu0"
        
        self._test_hparams = self._default_hparams().override_from_dict(test_hparams)
        self._model_path = model_path

        config_path = glob.glob(os.path.expanduser(model_path) + '/*.yaml')
        assert len(config_path) == 1, "there should be one yaml file with params inside model_path but instead {} were found!".format(len(config_path))
        config_path = config_path[0]

        with open(config_path) as config:
            params = yaml.load(config, Loader=yaml.SafeLoader)
            self._model_hparams = params['model']
            self._input_hparams = params['dataset']

        print('\n\n------------------------------------ LOADED PARAMS ------------------------------------')
        for k, v in self._model_hparams.items():
            print('{} --> {}'.format(k, v))
        for k, v in self._input_hparams.items():
            print('{} --> {}'.format(k, v))
        print('---------------------------------------------------------------------------------------\n\n')
        
        PredictionModel = get_model(self._model_hparams.pop('model'))
        self._model = PredictionModel(self._input_hparams, n_gpus, self._model_hparams.pop('graph_type'), False, self._model_hparams.pop('scope_name'))
        self._outputs = self._model.model_fn(self._build_inputs(), {}, tf.estimator.ModeKeys.PREDICT, self._model_hparams)

        self._sess = sess
        self._restored = False
    
    def _default_hparams(self):
        default_dict = {
            "run_batch_size": 200,
            'tile_context': True,
            'designated_pixel_count': 0
        }
        return HParams(**default_dict)
    
    def _build_inputs(self):
        B_pl = self._test_hparams.run_batch_size
        if self._test_hparams.tile_context:
            B_pl = 1
        
        context_frames = self._model_hparams['context_frames']
        assert context_frames > 1, "needs at least 1 context action (so 2 frames)"
        
        input_length = self._model_hparams['sequence_length'] - 1
        pad_len = input_length - context_frames
        
        height, width = self._input_hparams['img_size']
        self._images_pl = tf.placeholder(tf.float32, [B_pl, context_frames, height, width, 3])
        self._states_pl = tf.placeholder(tf.float32, [B_pl, context_frames, self._input_hparams['target_sdim']])
        self._context_actions_pl = tf.placeholder(tf.float32, [B_pl, context_frames - 1, self._input_hparams['target_adim']])
        self._actions_pl = tf.placeholder(tf.float32, [self._test_hparams.run_batch_size, pad_len + 1, self._input_hparams['target_adim']])

        if self._test_hparams.designated_pixel_count:
            self._pixel_dist_pl = tf.placeholder(tf.float32, [B_pl, context_frames, height, width, self._test_hparams.designated_pixel_count])
            pad = tf.zeros((B_pl, pad_len, height, width, self._test_hparams.designated_pixel_count), dtype=tf.float32)
            input_pixel_distributions = tf.concat((self._pixel_dist_pl, pad), axis=1)
            if self._test_hparams.tile_context:
                input_pixel_distributions = tf.tile(input_pixel_distributions, [self._test_hparams.run_batch_size, 1, 1, 1, 1])

        input_imgs = tf.concat((self._images_pl, tf.zeros((B_pl, pad_len, height, width, 3), dtype=tf.float32)), axis=1)
        input_states = tf.concat((self._states_pl, tf.zeros((B_pl, pad_len, self._input_hparams['target_sdim']), dtype=tf.float32)), axis=1)        
        if self._test_hparams.tile_context:
            input_states, context_actions = [tf.tile(tensor, [self._test_hparams.run_batch_size, 1, 1]) for tensor in [input_states, self._context_actions_pl]]
            input_imgs = tf.tile(input_imgs, [self._test_hparams.run_batch_size, 1, 1, 1, 1]) 
        else:
            context_actions = self._context_actions_pl
        
        input_actions = tf.concat((context_actions, self._actions_pl), axis=1)

        ret_dict = {'actions': input_actions, 'images': input_imgs, 'states': input_states}
        if self._test_hparams.designated_pixel_count:
            ret_dict['pixel_distributions'] =  input_pixel_distributions
        return ret_dict
    
    def predict(self, context_tensors, action_tensors):
        # assert self._restored, "must restore before testing can continue!"
        
        if self._test_hparams.tile_context:
            assert context_tensors['context_frames'].shape[1] == 1, "only one camera supported!"
            context_images = context_tensors['context_frames'][-self._model_hparams['context_frames']:, 0][None]
            context_actions = context_tensors['context_actions'][(1 - self._model_hparams['context_frames']):][None]
            context_states = context_tensors['context_states'][-self._model_hparams['context_frames']:][None]
        else:
            assert context_tensors['context_frames'].shape[2] == 1, "only one camera supported!"
            context_images = context_tensors['context_frames'][:, -self._model_hparams['context_frames']:, 0]
            context_actions = context_tensors['context_actions'][:, (1 - self._model_hparams['context_frames']):]
            context_states = context_tensors['context_states'][:, -self._model_hparams['context_frames']:]
        
        if self._test_hparams.designated_pixel_count and self._test_hparams.tile_context:
            context_distributions = context_tensors['context_pixel_distributions'][-self._model_hparams['context_frames']:, 0][None]
        elif self._test_hparams.designated_pixel_count:
            context_distributions = context_tensors['context_pixel_distributions'][:, -self._model_hparams['context_frames']:, 0]
        else: 
            context_distributions = None

        input_actions = action_tensors['actions']
        n_runs = int(math.ceil(input_actions.shape[0] / float(self._test_hparams.run_batch_size)))
        assert n_runs

        ret_dict = None
        for n in range(n_runs):
            selected_actions = input_actions[n * self._test_hparams.run_batch_size :(n + 1) * self._test_hparams.run_batch_size]
            if selected_actions.shape[0] < self._test_hparams.run_batch_size:
                pad = np.zeros((self._test_hparams.run_batch_size - selected_actions.shape[0], selected_actions.shape[1], selected_actions.shape[2]))
                padded_actions = np.concatenate((selected_actions, pad), axis=0)
            else:
                padded_actions = selected_actions
            
            run_t = self._feed(context_images, context_actions, context_states, context_distributions, padded_actions)
            
            for k in run_t.keys():
                run_t[k] = run_t[k][:selected_actions.shape[0]]

            if ret_dict is None:
                ret_dict = run_t
            else:
                for k, v in run_t.items():
                    ret_dict[k] = np.concatenate((ret_dict[k], v), axis=0)
        return ret_dict

    def _feed(self, context_images, context_actions, context_states, context_distributions, input_actions):
        if context_images.dtype == np.uint8:
            context_images = context_images.astype(np.float32) / 255
        
        feed_dict = {self._images_pl: context_images,
                        self._states_pl: context_states, 
                        self._context_actions_pl: context_actions, 
                        self._actions_pl: input_actions}

        if self._test_hparams.designated_pixel_count and context_distributions is None:
            height, width = self._input_hparams['img_size']
            context_distributions = np.zeros((self._test_hparams.batch_size, self._model_hparams['context_frames'], 
                                                height, width, self._test_hparams.designated_pixel_count), dtype=np.float32)
            context_distributions[:, :, 0, 0] = 1.0
            feed_dict[self._pixel_dist_pl] = context_distributions
        elif self._test_hparams.designated_pixel_count:
            feed_dict[self._pixel_dist_pl] = context_distributions

        return self._sess.run(self._outputs, feed_dict=feed_dict)
    
    def __call__(self, context_tensors, action_tensors):
        return self.predict(context_tensors, action_tensors)

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
    def sequence_length(self):
        return self._model_hparams['sequence_length']
    
    @property
    def n_context(self):
        return self._model_hparams['context_frames']

    @property
    def n_cam(self):
        return 1

    @property
    def img_size(self):
        return self._input_hparams['img_size']
