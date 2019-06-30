from robonet.datasets.variants.held_out_robot_dataset import HeldoutRobotDataset
from robonet.datasets.variants.annotation_benchmark_dataset import AnnotationBenchmarkDataset

class AnnotationHeldoutRobotDataset(HeldoutRobotDataset, AnnotationBenchmarkDataset):
    def train_val_filter(self, train_metadata, val_metadata):
        train_metadata, val_metadata = HeldoutRobotDataset.train_val_filter(self, train_metadata, val_metadata)
        train_metadata, val_metadata = AnnotationBenchmarkDataset.train_val_filter(self, train_metadata, val_metadata)
        return train_metadata, val_metadata
