from .hdf5_datasets.robonet_wrapper import RoboNetVideoDataset

def get_dataset_class(dataset):
    dataset_mappings = {
        'RoboNet': 'RoboNetVideoDataset'
    }
    dataset_class = dataset_mappings.get(dataset, dataset)
    dataset_class = globals().get(dataset_class)
    return dataset_class
