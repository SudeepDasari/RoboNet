from .util.metadata_helper import load_metadata


def get_dataset_class(name):
    if name == 'RoboNet':
        from .robonet_dataset import RoboNetDataset
        return RoboNetDataset
    elif name == 'AnnotatedRoboNet':
        from .variants.annotation_benchmark_dataset import AnnotationBenchmarkDataset
        return AnnotationBenchmarkDataset
    elif name == 'AnnotationHeldoutRobotDataset':
        from .variants.val_filter_dataset_variants import AnnotationHeldoutRobotDataset
        return AnnotationHeldoutRobotDataset
    elif name == 'HeldoutRobotDataset':
        from .variants.val_filter_dataset_variants import HeldoutRobotDataset
        return HeldoutRobotDataset
    else:
        raise NotImplementedError
