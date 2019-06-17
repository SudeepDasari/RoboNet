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
        raise NotImplementedError
