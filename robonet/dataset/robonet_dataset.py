from visual_mpc.datasets.hdf5_dataset import HDF5VideoDataset
from visual_mpc.datasets.save_util.filter_dataset import cached_filter_hdf5
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
            # should rework this section
            raise NotImplementedError
        else:
            chosen_files = []
            [chosen_files.extend(filtered_datasets[k]) for k in chosen_ncam]
            self._data_loaders = [HDF5VideoDataset(chosen_files, self._batch_size, self._dict_copy, append_path=self._files)]
    
    def _get(self, key, mode):
        ret_tensor = []
        for dataset in self._data_loaders:
            if key == 'images' and self._source_views:
                cam_images = tf.transpose(dataset[key, mode], [2, 0, 1, 3, 4, 5])
                chosen_cams = tf.gather(cam_images, self._source_views)
                ret_tensor.append(tf.transpose(cam_images, [1, 2, 0, 3, 4, 5]))
            else:
                ret_tensor.append(dataset[key, mode])
        return tf.concat(ret_tensor, axis=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="converts dataset from pkl format to hdf5")
    parser.add_argument('input_folder', type=str, help='folder containing hdf5 files')
    args = parser.parse_args()

    path = args.input_folder
    rn = RoboNetDataset(path, [16], {'filters': [{"metadata/bin_insert": "none"}]})
    images, states, actions = rn['images'], rn['states'], rn['actions']
