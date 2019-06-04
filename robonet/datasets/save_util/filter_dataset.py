import h5py
import pickle as pkl
import sys
if sys.version_info[0] == 2:
    import cPickle as pkl
else:
    import pickle as pkl
from tqdm import tqdm
import os
import hashlib
import numpy as np


class _StrHashDict:
    """
    "Hashable" python dictionary
    - result only valid so long as str() properly overriden
    """
    def __init__(self):
        self._dict = {}
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __setitem__(self, key, val):
        self._dict[key] = val
    
    def __hash__(self):
        key_hashes = []
        for k in sorted(self._dict.keys()):
            if isinstance(self._dict[k], np.ndarray):
                key_hashes.append((k, hash(self._dict[k].tostring())))
            else:
                key_hashes.append((k, hash(self._dict[k])))
        
        return hash(tuple(key_hashes))

    def __eq__(self, other):
        if not isinstance(other, _StrHashDict):
            return False
        if not set([k for k in self._dict.keys()]) == set([k for k in other._dict.keys()]):
            return False
        
        for k in self._dict.keys():
            if isinstance(self._dict[k], np.ndarray) and isinstance(other._dict[k], np.ndarray):
                if not all(self._dict[k] == other._dict[k]):
                    return False
            elif isinstance(self._dict, np.ndarray):
                return False
            elif not self._dict[k] == other._dict[k]:
                    return False
        return True
    
    def __str__(self):
        return str(self._dict)
    
    def keys(self):
        return self._dict.keys()
    
    def items(self):
        return self._dict.items()
    
    def get(self, key, default_value=None):
        return self._dict.get(key, default_value)


def filter_hdf5_datasets(dataset_files):
    distinct_datasets = {}
    print('filtering datasets.....')
    for  f in tqdm(dataset_files):
        file_metadata = _StrHashDict()
        with h5py.File(f, 'r') as hf:
            file_metadata['action_T'], file_metadata['adim'] = hf['policy']['actions'].shape
            file_metadata['state_T'], file_metadata['sdim'] = hf['env']['state'].shape
            file_metadata['ncam'] = hf['env'].attrs.get('n_cams', 0)
            file_metadata['img_T'] = min([len(hf['env']['cam{}_video'.format(i)]) for i in range(file_metadata['ncam'])])
            if 'frames' in hf['env']['cam0_video']:
               file_metadata['img_dim'] = hf['env']['cam0_video']['frames'].attrs['shape'][:2]
            else:
               file_metadata['img_dim'] = hf['env']['cam0_video']['frame0'].attrs['shape'][:2]
            if 'goal_reached' in hf['misc']:
                file_metadata['goal_reached'] = hf['misc']['goal_reached'][()]

            for k in hf['metadata'].attrs.keys():
                file_metadata['metadata/{}'.format(k)] = hf['metadata'].attrs[k]
        f_name = f.split('/')[-1]
        
        if file_metadata in distinct_datasets:
            distinct_datasets[file_metadata].add(f_name)
        else:
            f_set = set()
            f_set.add(f_name)
            distinct_datasets[file_metadata] = f_set
        
    return distinct_datasets


def cached_filter_hdf5(dataset_files, cache_file):
    fname_hash = hashlib.sha256('+'.join([d.split('/')[-1] for d in sorted(dataset_files)]).encode('utf-8')).hexdigest()
    file_exists, cached_filter = os.path.exists(cache_file), None

    if file_exists:
        try:
            saved_hash, filtered_datasets = pkl.load(open(cache_file, 'rb'))
            if saved_hash == fname_hash:
                return filtered_datasets
        except EOFError:
            pass
    
    print("Cache doesn't check out! Calculating may take a while")
    filtered_datasets = filter_hdf5_datasets(dataset_files)
    if sys.version_info[0] == 2:
        pkl.dump((fname_hash, filtered_datasets), open(cache_file, 'wb'))
    else:
        pkl.dump((fname_hash, filtered_datasets), open(cache_file, 'wb'), 2)
    
    return filtered_datasets
