import tensorflow as tf


class BaseGraph:
    def build_graph(self, inputs, hparams, scope_name='graph'):
        raise NotImplementedError

    @staticmethod
    def default_hparams():
        return {
            'sequence_length': 15,
            'context_frames': 2,
            'use_states': False
        }

    @property
    def vars(self):
        return tf.trainable_variables(self._scope_name)

