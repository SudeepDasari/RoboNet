from robonet.datasets.variants.val_filter_dataset import ValFilterDataset

class HeldoutRobotDataset(ValFilterDataset):
    """
    Use files from one held-out robot for testing and files from all other robots for training
    """
    @staticmethod
    def _get_default_hparams():
        parent_hparams = ValFilterDataset._get_default_hparams()
        parent_hparams.add_hparam('held_out_robot', '')
        return parent_hparams

    def train_val_filter(self, train_metadata, val_metadata):
        train_metadata = train_metadata[train_metadata['robot'] != self._hparams.held_out_robot]
        val_metadata = val_metadata[val_metadata['robot'] == self._hparams.held_out_robot]
        return train_metadata, val_metadata


