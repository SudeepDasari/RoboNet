import ray
from robonet.video_prediction.models import get_model_fn
import numpy as np
import json
from robonet.video_prediction.utils import tf_utils
import tensorflow as tf
from tensorflow.contrib.training import HParams


class VPredEvaluation(object):
    def __init__(self, model_hparams_path, test_hparams, n_gpus=1, first_gpu=0):
        assert n_gpus == 1, "multi gpu evaluation not yet written"
        assert first_gpu == 0, "only starts building at gpu0"
        
        self._test_hparams = self._default_hparams.override_from_dict(test_hparams)
        self._model_hparams = json.load(open(model_hparams_path, 'r'))

        graph_type = self._model_hparams.pop('graph_type')
        model_fn = get_model_fn(self._model_hparams.pop('model'))
        self._outputs = model_fn(n_gpus, graph_type, False, self._build_inputs(), None, tf.estimator.ModeKeys.PREDICT, self.model_hparams)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def _default_hparams(self):
        default_dict {
            "img_dims": [48, 64],
            "adim": 4,
            "sdim": 5
        }
        return HParams(**default_dict)
    
    def _build_inputs(self):
        context_frames = self._model_hparams['context_frames']
        input_length = self._model_hparams['sequence_length'] - 1
        pad_len = input_length - context_frames
        
        height, width = self._test_hparams.img_dims
        self._images_pl = tf.placeholder(tf.float32, [None, context_frames, height, width, 3])
        self._pixel_dist_pl = tf.placeholder(tf.float32, [None, context_frames, height, width, 2])
        self._states_pl = tf.placeholder(tf.float32, [None, context_frames, self._test_hparams.sdim])
        self._context_actions_pl = tf.placeholder(tf.float32, [None, context_frames, self._test_hparams.adim])
        self._actions_pl = tf.placeholder(tf.float32, [None, pad_len, self._test_hparams.adim])

        input_imgs = tf.concat((self._images_pl, tf.zeros((self._test_hparams.batch_size, pad_len, height, width, 3), dtype=tf.float32)), axis=1)
        input_pixel_distributions = tf.concat((self._pixel_dist_pl, tf.zeros((self._test_hparams.batch_size, pad_len, height, width, 2), dtype=tf.float32)), axis=1)
        input_actions = tf.concat((self._context_actions_pl, self._actions_pl, dtype=tf.float32)), axis=1)
        input_states = tf.concat((self._states_pl, tf.zeros((self._test_hparams.batch_size, pad_len, self._test_hparams.sdim), dtype=tf.float32)), axis=1)

        return {'actions': input_actions, 'pixel_distributions': input_pixel_distributions, 'images': input_imgs, 'states': input_states}
    
    def predict(context_tensors, action_tensors):
        context_images = context_tensors['context_frames']
        context_actions = context_tensors['context_frames']
        context_states = context_tensors['context_states']
        context_distributions = context_tensors.get('context_distributions', None)
        
        input_actions = action_tensors['actions']

        if context_distributions is None:
            height, width = self._test_hparams.img_dims
            context_distributions = np.zeros((self._test_hparams.batch_size, self._model_hparams['context_frames'], height, width, 2), dtype=np.float32)
            context_distributions[:, :, 0, 0] = 1.0
        return self.sess.run(self._outputs, feed_dict={self._images_pl: context_images, self._pixel_dist_pl: context_distributions, self._states_pl: context_states: 
                                                       self._context_actions_pl: context_actions, self._actions_pl: input_actions})
    
    def __call__(self, context_tensors, action_tensors):
        return self.predict(context_tensors, action_tensors)

    def restore(self, restore_path):
        checkpoints = [restore_path]
        # automatically skip global_step if more than one checkpoint is provided
        skip_global_step = len(checkpoints) > 1
        savers = []
        for checkpoint in checkpoints:
            print("creating restore saver from checkpoint %s" % checkpoint)
            saver, _ = tf_utils.get_checkpoint_restore_saver(checkpoint, skip_global_step=skip_global_step)
            savers.append(saver)
        restore_op = [saver.saver_def.restore_op_name for saver in savers]
        return self.sess.run(restore_op)
