import ray
from robonet.video_prediction.models import get_model_fn
import numpy as np
import json
from robonet.video_prediction.utils import tf_utils
import tensorflow as tf
from tensorflow.contrib.training import HParams
import os
import math


class VPredEvaluation(object):
    def __init__(self, model_hparams_path, test_hparams, n_gpus=1, first_gpu=0):
        assert n_gpus == 1, "multi gpu evaluation not yet written"
        assert first_gpu == 0, "only starts building at gpu0"
        
        self._test_hparams = self._default_hparams().override_from_dict(test_hparams)
        if model_hparams_path[:2] == '~/':
            model_hparams_path = os.path.expanduser(model_hparams_path)
        loaded_json = json.load(open(model_hparams_path, 'r'))
        if "checkpoints" in loaded_json:
            self._model_hparams = loaded_json['checkpoints'][0]["config"]['model_hparams']
            dataset_hparams = loaded_json['checkpoints'][0]["config"]['dataset_hparams']
            assert dataset_hparams.get('target_adim', 4)  == self._test_hparams.adim
            assert dataset_hparams.get('target_sdim', 5)  == self._test_hparams.sdim
        else:
            self._model_hparams = loaded_json
            print('no dataset hparams found - there is way to detect a/sdim mismatch errors')

        print('\n\n------------------------------------ LOADED PARAMS ------------------------------------')
        for k, v in self._model_hparams.items():
            print('{} --> {}'.format(k, v))
        print('---------------------------------------------------------------------------------------\n\n')
        graph_type = self._model_hparams.pop('graph_type')

        model_fn = get_model_fn(self._model_hparams.pop('model'))
        self._outputs = model_fn(n_gpus, graph_type, False, self._build_inputs(), None, tf.estimator.ModeKeys.PREDICT, self._model_hparams)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self._restored = False
    
    def _default_hparams(self):
        default_dict = {
            "run_batch_size": 200,
            "img_dims": [48, 64],
            "adim": 4,
            "sdim": 5,
            'designated_pixel_count': 0
        }
        return HParams(**default_dict)
    
    def _build_inputs(self):
        context_frames = self._model_hparams['context_frames']
        assert context_frames > 1, "needs at least 1 context action (so 2 frames)"
        
        input_length = self._model_hparams['sequence_length'] - 1
        pad_len = input_length - context_frames
        
        height, width = self._test_hparams.img_dims
        self._images_pl = tf.placeholder(tf.float32, [1, context_frames, height, width, 3])
        self._states_pl = tf.placeholder(tf.float32, [1, context_frames, self._test_hparams.sdim])
        self._context_actions_pl = tf.placeholder(tf.float32, [1, context_frames - 1, self._test_hparams.adim])
        self._actions_pl = tf.placeholder(tf.float32, [self._test_hparams.run_batch_size, pad_len + 1, self._test_hparams.adim])

        if self._test_hparams.designated_pixel_count:
            self._pixel_dist_pl = tf.placeholder(tf.float32, [1, context_frames, height, width, self._test_hparams.designated_pixel_count])
            pad = tf.zeros((1, pad_len, height, width, self._test_hparams.designated_pixel_count), dtype=tf.float32)
            input_pixel_distributions = tf.concat((self._pixel_dist_pl, pad), axis=1)
            input_pixel_distributions = tf.tile(input_pixel_distributions, [self._test_hparams.run_batch_size, 1, 1, 1, 1])

        input_imgs = tf.concat((self._images_pl, tf.zeros((1, pad_len, height, width, 3), dtype=tf.float32)), axis=1)
        input_imgs = tf.tile(input_imgs, [self._test_hparams.run_batch_size, 1, 1, 1, 1]) 

        input_states = tf.concat((self._states_pl, tf.zeros((1, pad_len, self._test_hparams.sdim), dtype=tf.float32)), axis=1)        
        input_states, context_actions = [tf.tile(tensor, [self._test_hparams.run_batch_size, 1, 1]) for tensor in [input_states, self._context_actions_pl]]
        input_actions = tf.concat((context_actions, self._actions_pl), axis=1)

        return {'actions': input_actions, 'pixel_distributions': input_pixel_distributions, 'images': input_imgs, 'states': input_states}
    
    def predict(self, context_tensors, action_tensors):
        assert self._restored, "must restore before testing can continue!"
        assert context_tensors['context_frames'].shape[1] == 1, "only one camera supported!"
        context_images = context_tensors['context_frames'][-self._model_hparams['context_frames']:, 0][None]
        context_actions = context_tensors['context_actions'][(1 - self._model_hparams['context_frames']):][None]
        context_states = context_tensors['context_states'][-self._model_hparams['context_frames']:][None]
        context_distributions = context_tensors.get('context_pixel_distributions', None)
        if self._test_hparams.designated_pixel_count:
            context_distributions = context_distributions[-self._model_hparams['context_frames']:, 0][None]
        
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
            height, width = self._test_hparams.img_dims
            context_distributions = np.zeros((self._test_hparams.batch_size, self._model_hparams['context_frames'], 
                                                height, width, self._test_hparams.designated_pixel_count), dtype=np.float32)
            context_distributions[:, :, 0, 0] = 1.0
            feed_dict[self._pixel_dist_pl] = context_distributions
        elif self._test_hparams.designated_pixel_count:
            feed_dict[self._pixel_dist_pl] = context_distributions

        return self.sess.run(self._outputs, feed_dict=feed_dict)
    
    def __call__(self, context_tensors, action_tensors):
        return self.predict(context_tensors, action_tensors)

    def restore(self, restore_path):
        if restore_path[:2] == '~/':
            restore_path = os.path.expanduser(restore_path)
        checkpoints = [restore_path]
        # automatically skip global_step if more than one checkpoint is provided
        skip_global_step = len(checkpoints) > 1
        savers = []
        for checkpoint in checkpoints:
            print("creating restore saver from checkpoint %s" % checkpoint)
            saver, _ = tf_utils.get_checkpoint_restore_saver(checkpoint, skip_global_step=skip_global_step)
            savers.append(saver)
        restore_op = [saver.saver_def.restore_op_name for saver in savers]
        self.sess.run(restore_op)
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
