def get_dataset_class(name):
    if name == 'RoboNet':
        from .robonet_dataset import RoboNetDataset
        return RoboNetDataset
    elif name == 'hdf5':
        from .hdf5_dataset import HDF5VideoDataset
        return HDF5VideoDataset
    else:
        raise NotImplementedError
