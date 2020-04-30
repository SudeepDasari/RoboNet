from .util.metadata_helper import load_metadata


def get_dataset_class(name):
    if name == 'RoboNetSawyer':
        from .robonet_sawyer_dataset import SawyerDataset
        return SawyerDataset
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
    elif name == 'TPU' or name == 'TFRecords':
        from .record_dataset import TFRecordVideoDataset
        return TFRecordVideoDataset
    elif name == 'robonet_sawyer' or name == 'sawyer':
        from .robonet_sawyer_dataset import SawyerDataset
    else:
        raise NotImplementedError
