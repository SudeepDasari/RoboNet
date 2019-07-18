from robonet.video_prediction.models import get_graph_class
from tensorflow.contrib.training import HParams
import itertools


class BaseModel(object):
    def __init__(self, data_loader_hparams, num_gpus, graph_type, tpu_mode=False):
        self._data_hparams = data_loader_hparams
        self._num_gpus = num_gpus
        self._graph_class = get_graph_class(graph_type)
        self._tpu_mode = tpu_mode
    
    def init_default_hparams(self, params):
        graph_params = self._graph_class.default_hparams()
        model_hparams = self._model_default_hparams()
        default_hparams = dict(itertools.chain(graph_params.items(), model_hparams.items()))
        self._hparams = HParams(**default_hparams).override_from_dict(params)
    
    def model_fn(self, inputs, targets, mode, params):
        self.init_default_hparams(params)
        return self._model_fn(inputs, targets, mode)
