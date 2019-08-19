import ray
from robonet.inverse_model.models import get_models
import numpy as np
import json
from robonet.video_prediction.utils import tf_utils
import tensorflow as tf
from tensorflow.contrib.training import HParams
import os
import math


class ActionInferenceInterface(object):
    def __init__(self, model_hparams_path, test_hparams, n_gpus=1, first_gpu=0):
        assert n_gpus == 1, "multi gpu evaluation not yet written"
        assert first_gpu == 0, "only starts building at gpu0"
        
        self._test_hparams = self._default_hparams().override_from_dict(test_hparams)

        model_hparams_path = os.path.expanduser(model_hparams_path)
        loaded_json = json.load(open(model_hparams_path, 'r'))
        if "checkpoints" in loaded_json:
            self._model_hparams = loaded_json['checkpoints'][0]["config"]['model_hparams']
            dataset_hparams = loaded_json['checkpoints'][0]["config"]['loader_hparams']
            assert dataset_hparams.get('target_adim', 4)  == self._test_hparams.adim
        else:
            self._model_hparams = loaded_json
            dataset_hparams = {}
            print('no dataset hparams found - there is no way to detect adim mismatch errors')

        print('\n\n------------------------------------ LOADED PARAMS ------------------------------------')
        for k, v in self._model_hparams.items():
            print('{} --> {}'.format(k, v))
        print('---------------------------------------------------------------------------------------\n\n')
        
        PredictionModel = get_models(self._model_hparams.pop('model'))
        self._model_hparams['vgg_path'] = os.path.expanduser(self._test_hparams.vgg_path)
        self._model = model = PredictionModel(dataset_hparams, n_gpus, self._model_hparams['graph_type'], False)
        inputs, targets = self._build_input_targets()
        self._pred_act= model.model_fn(inputs, targets, tf.estimator.ModeKeys.PREDICT, self._model_hparams)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self._restored = False
    
    def _default_hparams(self):
        default_dict = {
            "img_dims": [48, 64],
            "adim": 4,
            "load_T": 10,
            "vgg_path": "~/"                # vgg19.npy should be in vgg_path folder (aka vgg_path = /path/to/folder/containing/weights/)
        }
        return HParams(**default_dict)
    
    def _build_input_targets(self):
        height, width = self._test_hparams.img_dims
        self._images_pl = tf.placeholder(tf.float32, [1, 2, height, width, 3])
        return {'adim': self._test_hparams.adim, 'T': self._test_hparams.load_T - 1}, {'images': self._images_pl}
    
    def predict(self, start_image, goal_image):
        assert self._restored
        start_goal_image = np.concatenate((start_image[None][None], goal_image[None][None]), axis=1)
        return self.sess.run(self._pred_act, feed_dict={self._images_pl: start_goal_image})
    
    def __call__(self, start_image, goal_image):
        return self.predict(start_image, goal_image)

    def restore(self, restore_path):
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


if __name__ == '__main__':
    test_hparams = {"load_T": 3}                                                                  # test hparams overrides, make sure you set load_T to the value used during training
    inference_object = ActionInferenceInterface('~/model_test/hparams.json', test_hparams)        # experiment_state...json file from ray
    inference_object.restore('~/model_test/checkpoint_100000/model-100000')

    start_image = np.zeros((48, 64, 3))
    goal_image = np.zeros((48, 64, 3))
    import pdb; pdb.set_trace()
    print(inference_object(start_image, goal_image))