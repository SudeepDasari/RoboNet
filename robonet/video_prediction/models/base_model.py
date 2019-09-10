from robonet.video_prediction.models import get_graph_class
from tensorflow.contrib.training import HParams
import itertools
import copy


class BaseModel(object):
    def __init__(self, data_loader_hparams, num_gpus, graph_type, tpu_mode=False, graph_scope='vpred_model'):
        self._data_hparams = data_loader_hparams
        self._num_gpus = num_gpus
        self._graph_class = self._get_graph(graph_type)
        self._tpu_mode = tpu_mode
        self._graph_scope = graph_scope
    
    def _get_graph(self, graph_type):
        return get_graph_class(graph_type)

    def init_default_hparams(self, params):
        graph_params = self._graph_class.default_hparams()
        model_hparams = self._model_default_hparams()
        default_hparams = dict(itertools.chain(graph_params.items(), model_hparams.items()))

        params = copy.deepcopy(params)
        if self._tpu_mode:
            self._summary_dir = params.pop('summary_dir')
            self._summary_queue_len = params.pop('summary_queue_len')
            self._image_summary_freq = params.pop('image_summary_freq')

        self._use_tpu = params.pop('use_tpu', None)
        for k in list(params.keys()):
            if k not in default_hparams:
                params.pop(k)

        self._hparams = HParams(**default_hparams).override_from_dict(params)
        self._hparams.use_tpu = self._use_tpu
    
    def model_fn(self, features, labels, mode, params):
        self.init_default_hparams(params)
        return self._model_fn(features, labels, mode)

    def _model_default_hparams(self):
        raise NotImplementedError
    
    def _model_fn(self, inputs, targets, mode):
        raise NotImplementedError
