from .util.metadata_helper import load_metadata


def get_dataset_class(name):
    if name == 'RoboNet':
        from .robonet_dataset import RoboNetDataset
        return RoboNetDataset
    else:
        raise NotImplementedError
