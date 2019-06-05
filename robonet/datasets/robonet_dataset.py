from robonet.datasets.hdf5_dataset import HDF5VideoDataset
from robonet.datasets.save_util.filter_dataset import cached_filter_hdf5
import copy
import random
import tensorflow as tf


def _check_filter(filter_dict, key_dict):
    for k in filter_dict:
        if filter_dict[k] != key_dict.get(k, None):
            return False
    return True


def _create_assignments(batch_size, dataset_sizes):
    assert all([isinstance(d, int) for d in dataset_sizes])
    assert all([d > 0 for d in dataset_sizes])
    number_chosen = sum(dataset_sizes)

    batch_assignments = [max(int(batch_size / float(number_chosen) * float(d)), 1) for d in dataset_sizes]
    diff = batch_size - sum(batch_assignments)
    assert diff >= 0                            # due to truncation and batch_size > len(dataset_sizes), this should never happen
    for _ in range(diff):
        batch_assignments[random.randint(0, len(batch_assignments) - 1)] += 1
    return batch_assignments


def _dataset_printout(dataset_attr, batch_size):
    print('\n\n--------------------------------------------------------------------------')
    print('Creating sub dataset with batch size:', batch_size)
    for k, v in dataset_attr.items():
        if k == 'img_dim':
            print('{}:  {} \t\t\t (before resize)'.format(k, v))
        else:
            print('{}:  {}'.format(k, v))
    print('--------------------------------------------------------------------------')

    
class RoboNetDataset(HDF5VideoDataset):
    def __init__(self, directory, sub_batch_sizes, hparams_dict=dict()):
        assert isinstance(directory, str), "must pass directory path for this value"

        hdf5_params = copy.deepcopy(hparams_dict)
        self._filters = hdf5_params.pop('filters', [])
        self._sub_batch_sizes = sub_batch_sizes
        self._source_views = hdf5_params.pop('source_views', [])
        self._ncams = hdf5_params.pop('ncams', -1)
        self._dict_copy = hdf5_params

        if self._filters:
            assert isinstance(self._sub_batch_sizes, list)
            batch_size = sum(self._sub_batch_sizes)
        elif isinstance(self._sub_batch_sizes, list):
            print('WARNING: filters not set but sub batch sizes is a list!')
            batch_size = sum(self._sub_batch_sizes)
        else:
            batch_size = sub_batch_sizes

        super(RoboNetDataset, self).__init__(directory, batch_size, hdf5_params)

    def _init_dataset(self):
        assert isinstance(self._files, str), "must pass a folder path into this reader!"
        dataset_files = self._get_filenames()
        assert len(dataset_files) > 0, "couldn't find dataset at {}".format(self._files)

        filtered_datasets = cached_filter_hdf5(dataset_files, '{}/filter_cache.pkl'.format(self._files))
        # in general should iterate on the filtering scheme here
        filt_by_ncam = {}
        for k in filtered_datasets.keys():
            k_ncams = k['ncam']
            cam_list = filt_by_ncam.get(k_ncams, [])
            cam_list.append(k)
            filt_by_ncam[k_ncams] = cam_list
        
        if self._ncams == -1:
            self._ncams = max(filt_by_ncam.keys())
            print("loading datasets with {} cameras!".format(self._ncams))
        elif self._ncams not in filt_by_ncam:
            raise ValueError("no datasets with {} cameras".format(self._ncams))
        
        if self._source_views:
            assert self._ncams >= max(self._source_views)
        chosen_ncam = filt_by_ncam[self._ncams]
        
        dataset_batches = {}
        if self._filters:
            chosen_files = []
            for f in self._filters:
                [chosen_files.extend(filtered_datasets[k]) for k in chosen_ncams if _check_filter(k, f)]
            self._data_loader = HDF5VideoDataset(chosen_files, self._batch_size, self._dict_copy, append_path=self._files)
        else:
            chosen_files = []
            [chosen_files.extend(filtered_datasets[k]) for k in chosen_ncam]
            self._data_loader = HDF5VideoDataset(chosen_files, self._batch_size, self._dict_copy, append_path=self._files)
    
    def _get(self, key, mode):
        return self._data_loader[key, mode]

    def make_input_targets(self, n_frames, n_context, mode, img_dtype=tf.float32):
        return self._data_loader.make_input_targets(n_frames, n_context, mode, img_dtype)
    
    @property
    def num_examples_per_epoch(self):
        return self._data_loader.num_examples_per_epoch

if __name__ == '__main__':
    import imageio
    import argparse
    import time
    import numpy as np


    parser = argparse.ArgumentParser(description="converts dataset from pkl format to hdf5")
    parser.add_argument('input_folder', type=str, help='folder containing hdf5 files')
    parser.add_argument('--N', type=int, help='number of timing tests to run', default=10)
    parser.add_argument('--debug_gif', type=str, help='saves debug gif at given path if desired', default='')
    args = parser.parse_args()

    path = args.input_folder
    rn = RoboNetDataset(path, [16])
    images, states, actions = rn['images'], rn['states'], rn['actions']

    s = tf.Session()
    imgs = s.run(images)
    print('images shape', imgs.shape)

    if args.N:
        start = time.time()
        for i in range(args.N):
            b_start = time.time()
            imgs = s.run(images)
            print('load {} was {} seconds'.format(i, time.time() - b_start))
        end = time.time()
        print('loading took {} seconds on average!'.format((end - start) / float(args.N)))
    if args.debug_gif:
        path = args.debug_gif + '.gif'
        writer = imageio.get_writer(path)
        [writer.append_data(imgs[0, t, 0].astype(np.uint8)) for t in range(imgs.shape[1])]
        writer.close()
