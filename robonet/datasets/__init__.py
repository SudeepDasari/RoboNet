from .util.metadata_helper import load_metadata
from .util.tensor_multiplexer import multiplex_train_val_test, MultiplexedTensors


def get_dataset_class(name):
    if name == 'RoboNet':
        from .robonet_dataset import RoboNetDataset
        return RoboNetDataset
    else:
        raise NotImplementedError
