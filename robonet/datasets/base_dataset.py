import os
import glob
import copy
from .util.metadata_helper import load_metadata, MetaDataContainer
import random
import numpy as np
from torch.utils.data import IterableDataset, DataLoader


BATCH_SHUFFLE_RNG = 5011757766786901527
class BaseVideoDataset(IterableDataset):
    def __init__(self, batch_size, dataset_files_or_metadata, mode='train', hparams=dict()):
        assert isinstance(batch_size, int), "batch_size must be an integer"
        self._batch_size = batch_size

        if isinstance(dataset_files_or_metadata, str):
            self._metadata = [load_metadata(dataset_files_or_metadata)]
        elif isinstance(dataset_files_or_metadata, MetaDataContainer):
            self._metadata = [dataset_files_or_metadata]
        elif isinstance(dataset_files_or_metadata, (list, tuple)):
            self._metadata = []
            for d in dataset_files_or_metadata:
                assert isinstance(d, (str, MetaDataContainer)), "potential dataset must be folder containing files or meta-data instance"
                if isinstance(d, str):
                    self._metadata.append(load_metadata(d))
                else:
                    self._metadata.append(d)

        # initialize hparams and store metadata_frame
        self._hparams = self._get_default_hparams()
        self._hparams.update(hparams)

        #initialize dataset
        assert mode in self.modes
        self._mode = mode
        self._init_rng()
        self._len = self._init_dataset()
        print('loaded {} files'.format(self._len))

    def _init_dataset(self):
        raise NotImplementedError

    def _init_rng(self):
        seed = self._hparams.get('RNG', None)
        if self._mode == 'train' and  self._hparams['use_random_train_seed']:
            seed = None
        if seed is not None:
            seed += [i for i, m in enumerate(self.modes) if m == self._mode][0]
        
        self._file_shuffle = random.Random(BATCH_SHUFFLE_RNG)
        self._random_generator = random.Random(seed)
        self._np_rng = np.random.RandomState(self._random_generator.getrandbits(32))
    
    @staticmethod
    def _get_default_hparams():
        default_dict = {
            'RNG': 11381294392481135266,
            'use_random_train_seed': False
        }
        return default_dict

    @property
    def hparams(self):
        return copy.deepcopy(self._hparams)
    
    def __len__(self):
        return self._len

    @property
    def modes(self):
        return ['train', 'val', 'test']

    def make_dataloader(self, pin_memory=True):
        return DataLoader(self, batch_size=None, num_workers=0, pin_memory=pin_memory)
