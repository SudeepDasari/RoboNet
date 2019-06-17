import tensorflow as tf
from collections import OrderedDict


def multiplex_train_val_test(dataset, key_name, train_cond=None):
    if train_cond is None:
        _train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
    else:
        _train_cond = train_cond

    tensors = [dataset[key_name, m] for m in ['train', 'val', 'test']]

    # returns tensors[i]
    val_test_tensor = tf.cond(_train_cond < 2, lambda: tensors[1], lambda: tensors[2])
    tensor = tf.cond(_train_cond < 1, lambda: tensors[0], lambda: val_test_tensor)
    
    if train_cond is None:
        return tensor, _train_cond
    return tensor


class MultiplexedTensors:
    def __init__(self, dataset, tensor_names):
        self._train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
        self._tensor_dict = OrderedDict()
        for t in tensor_names:
            self._tensor_dict[t] = multiplex_train_val_test(dataset, t, self._train_cond)

    def __getitem__(self, key):
        return self._tensor_dict[key]

    @property
    def dict(self):
        return self._tensor_dict
    
    @property
    def train(self):
        return {self._train_cond: 0}
    
    @property
    def val(self):
        return {self._train_cond: 1}
    
    @property
    def test(self):
        return {self._train_cond: 2}
