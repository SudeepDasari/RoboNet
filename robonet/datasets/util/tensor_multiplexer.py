import tensorflow as tf
from collections import OrderedDict


def multiplex_tensors(datasets, key_name, train_cond=None):
    if train_cond is None:
        _train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
    else:
        _train_cond = train_cond

    tensors = [datasets[m][key_name, m] for m in datasets.keys()]
    assert len(tensors), "can't multiplex across no modes!"

    if len(tensors) == 1:
        if train_cond is None:
            return tensors[0], _train_cond
        return tensors[0]
    
    top_tensor = tensors[-1]
    for ind in range(len(tensors) - 1, 0, -1):
        top_tensor = tf.cond(_train_cond < ind, lambda: tensors[ind - 1], lambda: top_tensor)

    if train_cond is None:
        return top_tensor, _train_cond
    return top_tensor


class MultiplexedTensors:
    def __init__(self, dataset, tensor_names):
        self._mode_ind = OrderedDict()
        self._datasets = OrderedDict()
        if isinstance(dataset, (list, tuple)):
            assert len(dataset) == len(dataset[0].modes), "length of dataset list must match number of modes"
            for d in dataset[1:]:
                assert set(d.modes) == set(dataset[0].modes), "all datasets must have same modes"

            for i, k in enumerate(dataset[0].modes):
                self._mode_ind[k] = i
                self._datasets[k] = dataset[i]
            self._unique_datasets = dataset
        else:
            for i, k in enumerate(dataset.modes):
                self._mode_ind[k] = i
                self._datasets[k] = dataset
            self._unique_datasets = [dataset]
        
        self._train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
        self._tensor_dict = OrderedDict()
        for t in tensor_names:
            self._tensor_dict[t] = multiplex_tensors(self._datasets, t, self._train_cond)

    def __getitem__(self, key):
        return self._tensor_dict[key]

    @property
    def dict(self):
        return self._tensor_dict
    
    def get_feed_dict(self, mode):
        assert isinstance(mode, str) 
        assert mode in self._mode_ind, "{} not supported! Modes are {}".foramt(mode, self._mode_ind.keys())
        dataset_feed = self._datasets[mode].build_feed_dict(mode)

        if len(self._unique_datasets) > 1:
            for d in self._unique_datasets:
                if d is not self._datasets[mode]:
                    dataset_feed.update(d.get_null_dict())

        dataset_feed[self._train_cond] = self._mode_ind[mode]
        return dataset_feed

    @property
    def modes(self):
        return list(self._mode_ind.keys())
