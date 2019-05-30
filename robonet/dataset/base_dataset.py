import os
from tensorflow.contrib.training import HParams
import glob
import copy


class BaseVideoDataset(object):
    MODES = ['train', 'test', 'val']

    def __init__(self, files, batch_size, hparams_dict=dict(), append_path=''):
        if isinstance(files, str):
            if not os.path.exists(files):
                raise FileNotFoundError('Base directory {} does not exist'.format(files))
            self._files = append_path + files
        else:
            assert isinstance(files, list), "must be list of string"
            self._files = [append_path + f for f in files]
        
        self._batch_size = batch_size

        # read dataset manifest and initialize hparams
        self._hparams = self._get_default_hparams().override_from_dict(hparams_dict)
        
        #initialize dataset class
        self._init_dataset()

    def _init_dataset(self):
        raise NotImplementedError

    @staticmethod
    def _get_default_hparams():
        default_dict = {
                        'num_epochs': None,
                        'buffer_size': 512
                        }
        return HParams(**default_dict)

    def _get_filenames(self):
        if isinstance(self._files, str):
            return glob.glob(self._files + '/*.hdf5')
        elif isinstance(self._files, (tuple, list)):
            return self._files
        else:
            raise ValueError
    
    def get(self, key, mode='train'):
        if mode not in self.MODES:
            raise ValueError('Mode {} not valid! Dataset has following modes: {}'.format(mode, self.MODES))
        return self._get(key, mode)
    
    def _get(self, key, mode):
        raise NotImplementedError

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) != 2:
                raise KeyError('Index should be in format: [Key, Mode] or [Key] (assumes default train mode)')
            key, mode = item
            return self.get(key, mode)

        return self.get(item)

    @property
    def batch_size(self):
        return self._batch_size
