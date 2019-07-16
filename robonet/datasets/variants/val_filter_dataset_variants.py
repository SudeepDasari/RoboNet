from robonet.datasets.robonet_dataset import RoboNetDataset
from tensorflow.contrib.training.python.training.hparam import HParams
from robonet.datasets.variants.annotation_benchmark_dataset import AnnotationBenchmarkDataset
import pdb


"""
Should perhaps update these to work with new API
"""
class ValFilterDataset(RoboNetDataset):
    """
    Separates files that have annotations and those which don't
        - files with annotations are loaded as validation files
        - all others are loaded as train/test
    """

    def _split_files(self, metadata):
        train_metadata, val_metadata = self.train_val_filter(metadata, metadata)

        train_files, test_files, val_files = [], [], []
        train_test_files = train_metadata.files
        val_files = val_metadata.files
        [self.rng.shuffle(files) for files in [train_test_files, val_files]]
        train_pivot = int(len(train_test_files) * self._hparams.splits[0])
        if self._hparams.splits[0]:
            train_files = train_test_files[:train_pivot]
        if self._hparams.splits[1]:
            val_files = val_files
        if self._hparams.splits[2]:
            test_files = train_test_files[train_pivot:]
        return train_files, val_files, test_files

    def train_val_filter(self, train_metadata, val_metadata):
        """
        :param metadata:
        :return: train_metadata, val_metadata
        """
        raise NotImplementedError

class HeldoutRobotDataset(ValFilterDataset):
    """
    Use files from one held-out robot for testing and files from all other robots for training
    """
    @staticmethod
    def _get_default_hparams(parent_hparams=None):
        if parent_hparams is None:
            parent_hparams = ValFilterDataset._get_default_hparams()
        parent_hparams.add_hparam('held_out_robot', '')
        return parent_hparams

    def train_val_filter(self, train_metadata, val_metadata):
        train_metadata = train_metadata[train_metadata['robot'] != self._hparams.held_out_robot]
        val_metadata = val_metadata[val_metadata['robot'] == self._hparams.held_out_robot]
        print('after filtering robots: number of trainfiles {} number of val files {}'.format(len(train_metadata.files), len(val_metadata.files)))
        return train_metadata, val_metadata


class AnnotationHeldoutRobotDataset(HeldoutRobotDataset, AnnotationBenchmarkDataset):

    @staticmethod
    def _get_default_hparams():
        combined_params = RoboNetDataset._get_default_hparams()
        combined_params = HeldoutRobotDataset._get_default_hparams(combined_params)
        combined_params = AnnotationBenchmarkDataset._get_default_hparams(combined_params)
        return combined_params

    def train_val_filter(self, train_metadata, val_metadata):
        print('before filtering: number of trainfiles {} number of val files {}'.format(len(train_metadata.files), len(val_metadata.files)))
        train_metadata, val_metadata = HeldoutRobotDataset.train_val_filter(self, train_metadata, val_metadata)
        train_metadata, val_metadata = AnnotationBenchmarkDataset.train_val_filter(self, train_metadata, val_metadata)
        return train_metadata, val_metadata