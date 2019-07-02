from .util.metadata_helper import load_metadata
from .util.tensor_multiplexer import multiplex_train_val_test, MultiplexedTensors


def get_dataset_class(name):
    if name == 'RoboNet':
        from .robonet_dataset import RoboNetDataset
        return RoboNetDataset
    elif name == 'AnnotatedRoboNet':
        from robonet.datasets.variants.annotation_benchmark_dataset import AnnotationBenchmarkDataset
        return AnnotationBenchmarkDataset
    elif name == 'AnnotationHeldoutRobotDataset':
        from .variants.val_filter_dataset_variants import AnnotationHeldoutRobotDataset
        return AnnotationHeldoutRobotDataset
    elif name == 'HeldoutRobotDataset':
        from .variants.val_filter_dataset_variants import HeldoutRobotDataset
        return HeldoutRobotDataset
    else:
        raise NotImplementedError
