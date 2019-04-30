try:
    from visual_mpc.datasets import RoboNetDataset
except:
    print('did you install visual_foresight repo?')
import json
import os
import random
import tensorflow as tf
from collections import OrderedDict


def _slice_helper(tensor, start_i, N, axis):
    shape = tensor.get_shape().as_list()
    assert 0 <= axis < len(shape), "bad axis!"

    starts = [0 for _ in shape]
    ends = shape
    starts[axis], ends[axis] = start_i, N

    return tf.slice(tensor, starts, ends)

class RoboNetVideoDataset:
    # maybe fix the inheritance here more later?
    def __init__(self, files, config_path, batch_size):
        self._n_files = len(files)
        assert os.path.exists(config_path) and os.path.isfile(config_path), 'dataset params path incorrect!'
        self._config = {}
        with open(config_path) as f:
            self._config.update(json.loads(f.read()))
        self._use_states = self._config.pop('use_states', False)
        
        if "filters" in self._config:
            assert "sub_batches" in self._config, "filter options requires specified sub-batch proportions!"
            sub_batches = self._config.pop('sub_batches')
            assert len(sub_batches) == len(self._config['filters']), 'filter and sub batches must be same length!'
            assert sum(sub_batches) == 1.0 and \
                                all([0 <= x <= 1 for x in sub_batches]), 'sub-batches must be probability measure!'

            batches = [int(float(batch_size) * x) for x in sub_batches]
            batch_sum = sum(batches)
            assert batch_sum <= batch_size
            if batch_sum < batch_size:
                diff = batch_size - batch_sum
                for _ in range(diff):
                    batches[random.randint(0, len(batches) - 1)] += 1
            for i, b in enumerate(batches):
                print('filter {} has batch size {}'.format(i, b))
        else:
            batches = [batch_size]
        
        self._dataset = RoboNetDataset(files, batches, self._config)
        self._rand_start = None
        self._rand_cam = None
    
    def __get__(key, mode='train'):
        return self._dataset[key, mode]

    def make_input_targets(self, n_frames, n_context, mode):
        assert n_frames > 0
        if self._rand_start is None:
            img_T = self._dataset['images', mode].get_shape().as_list()[1]
            self._rand_start = tf.random_uniform((), maxval=img_T - n_frames + 1, dtype=tf.int32)

        if self._rand_cam is None:
            n_cam =  self._dataset['images', mode].get_shape().as_list()[2]
            self._rand_cam = tf.random_uniform((), maxval=n_cam, dtype=tf.int32)
        
        inputs = OrderedDict()
        inputs['images'] = _slice_helper(self._dataset['images', mode], self._rand_start, n_frames, 1)[:, :, self._rand_cam]
        if self._use_states:
            inputs['states'] = _slice_helper(self._dataset['states', mode], self._rand_start, n_frames, 1)
        inputs['actions'] = _slice_helper(self._dataset['actions', mode], self._rand_start, n_frames-1, 1)
        
        targets = _slice_helper(self._dataset['images', mode], self._rand_start + n_context, n_frames - n_context, 1)[:, :, self._rand_cam]
        return inputs, targets
    
    @property
    def hparams(self):
        return self._dataset._hparams

    def num_examples_per_epoch(self):
        return int(self._n_files * self.hparams.splits[0])