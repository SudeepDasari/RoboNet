import tensorflow as tf
from collections import OrderedDict


def multiplex_tensors(dataset, key_name, train_cond=None):
    if train_cond is None:
        _train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
    else:
        _train_cond = train_cond

    tensors = [dataset[key_name, m] for m in dataset.modes]
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
        self._mode_ind = {}
        for i, k in enumerate(dataset.modes):
            self._mode_ind[k] = i
        
        self._train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
        self._tensor_dict = OrderedDict()
        for t in tensor_names:
            self._tensor_dict[t] = multiplex_tensors(dataset, t, self._train_cond)

    def __getitem__(self, key):
        return self._tensor_dict[key]

    @property
    def dict(self):
        return self._tensor_dict
    
    def get_feed_dict(self, mode):
        if isinstance(mode, int):
            assert 0 <= mode < len(self._mode_ind.keys()), "mode_index must be in range 0 to len(modes) - 1"
            return {self._train_cond: mode}
        
        assert isinstance(mode, str) 
        assert mode in self._mode_ind, "{} not supported! Modes are {}".foramt(mode, self._mode_ind.keys())
        return {self._train_cond: self._mode_ind[mode]}

    @property
    def modes(self):
        return list(self._mode_ind.keys())
