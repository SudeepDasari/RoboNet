from robonet.datasets.base_dataset import BaseVideoDataset
from robonet.datasets.util.hdf5_loader import load_data, default_loader_hparams
from robonet.datasets.util.dataset_utils import color_augment, split_train_val_test
import numpy as np
import copy
import os
import torch
import random


class RoboNetDataset(BaseVideoDataset):
    def __init__(self, dataset_files_or_metadata, mode='train', hparams=dict()):
        hparams['source_probs'] = hparams.pop('source_probs', None)
        if hparams['source_probs']:
            assert len(hparams['source_probs']) == len(dataset_files_or_metadata), "requires exactly one source probability per data source"
            assert all([0 <= x <= 1 for x in hparams['source_probs']]) and sum(hparams['source_probs']) == 1, "not valid probability distribution"
        super(RoboNetDataset, self).__init__(dataset_files_or_metadata, mode, hparams)

    def _init_dataset(self):
        # check batch_size
        assert self._hparams['load_T'] >=0, "load_T should be non-negative!"

        # smallest max step length of all dataset sources 
        min_steps = min([min(min(m.frame['img_T']), min(m.frame['state_T'])) for m in self._metadata])
        if not self._hparams['load_T']:
            self._hparams['load_T'] = min_steps
        else:
            assert self._hparams['load_T'] <= min_steps, "ask to load {} steps but some records only have {}!".format(self._hparams['min_T'], min_steps)

        mode_index = [i for i, m in enumerate(self.modes) if m == self._mode][0]
        sources, sources_metadata = [], []
        for m_ind, metadata in enumerate(self._metadata):
            files_per_source = self._split_files(m_ind, metadata)
            assert len(files_per_source) == len(self.modes), "files should be split into {} sets (it's okay if sets are empty)".format(len(self.modes))
            sources.append(files_per_source[mode_index])
            sources_metadata.append(metadata)

        self._files = []
        if self._hparams['source_probs'] is not None and len(sources) > 1:
            total_files = max([len(s) / sp for s, sp in zip(sources, self._hparams['source_probs'])])
            for s, p, sm in zip(sources, self._hparams['source_probs'], sources_metadata):
                n_samples = int(np.ceil(total_files * p))
                if n_samples > len(s):
                    samples = np.concatenate((np.arange(len(s)), np.random.choice(len(sources), size=n_samples - len(s), replace=n_samples-len(s)>len(s))))
                else:
                    samples = np.random.choice(len(s), size=n_samples, replace=False)
                self._files.extend([(s[i], sm) for i in samples])
        else:
            for source, metadata in zip(sources, sources_metadata):
                self._files.extend([(s, metadata) for s in source])
        
        self._random_generator.shuffle(self._files)
        return len(self._files)

    def _split_files(self, source_number, metadata):
        if self._hparams['train_ex_per_source'] != [-1]:
            return split_train_val_test(metadata, train_ex=self._hparams['train_ex_per_source'][source_number], rng=self._file_shuffle)
        return split_train_val_test(metadata, splits=self._hparams['splits'], rng=self._file_shuffle)

    @staticmethod
    def _get_default_hparams():
        default_dict = {
            'RNG': 11381294392481135266,             # random seed to initialize data loader rng
            'use_random_train_seed': False,          # use a random seed for training objects 
            'splits': (0.9, 0.05, 0.05),             # train, val, test
            'num_epochs': None,                      # maximum number of epochs (None if iterate forever)
            'ret_fnames': False,                     # return file names of loaded hdf5 record
            'all_modes_max_workers': True,           # use multi-threaded workers regardless of the mode
            'load_random_cam': True,                 # load a random camera for each example
            'color_augmentation':0.0,                # std of color augmentation (set to 0 for no augmentations)
            'train_ex_per_source': [-1],             # list of train_examples per source (set to [-1] to rely on splits only)
            'MAX_RESETS': 10                         # maximum number of pool resets before error is raised in main thread
        }
        for k, v in default_loader_hparams().items():
            default_dict[k] = v
        
        return default_dict

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rng = random.Random(self._random_generator.getrandbits(32))

        file_hparams = copy.deepcopy(self._hparams)
        if worker_info is None:
            assigned_files = self._files
        else:
            per_worker = int(np.ceil(len(self._files) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self._files))
            assigned_files = self._files[iter_start:iter_end]
        
        for f_name, metadata in assigned_files:
            f_metadata = metadata.get_file_metadata(f_name)
            if self._hparams['load_random_cam']:
                file_hparams['cams_to_load'] = [rng.randint(0, int(f_metadata['ncam']) - 1)]
            yield load_data(f_name, f_metadata, file_hparams, rng.getrandbits(32))


def _timing_test(N, loader):
    import time
    import random

    timings = []
    i = 0 
    start = time.time()
    for _ in loader:
        run_time = time.time() - start
        timings.append(run_time)
        print('run {}, took {} seconds'.format(i, run_time))
        i += 1
        start = time.time()

        if i >= N:
            break

    if timings:
        print('train runs took on average {} seconds'.format(sum(timings) / len(timings)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="calculates or loads meta_data frame")
    parser.add_argument('path', help='path to files containing hdf5 dataset')
    parser.add_argument('--robots', type=str, nargs='+', default=None, help='will construct a dataset with batches split across given robots')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for test loader (should be even for non-time test demo to work)')
    parser.add_argument('--mode', type=str, default='train', help='mode to grab data from')
    parser.add_argument('--time_test', type=int, default=0, help='if value provided will run N timing tests')
    parser.add_argument('--load_steps', type=int, default=0, help='if value is provided will load <load_steps> steps')
    args = parser.parse_args()

    hparams = {'RNG': 0, 'ret_fnames': True, 'load_T': args.load_steps, 'action_mismatch': 3, 'state_mismatch': 3, 'splits':[0.8, 0.1, 0.1]}
    if args.robots:
        from robonet.datasets import load_metadata
        meta_data = load_metadata(args.path)
        hparams['source_probs'] = [1.0 / len(args.robots) for _ in args.robots]
        dataset = RoboNetDataset([meta_data[meta_data['robot'] == r] for r in args.robots], hparams=hparams)
    else:
        dataset = RoboNetDataset(args.path, hparams=hparams)
    
    loader = dataset.make_dataloader(args.batch_size)
    if args.time_test:
        _timing_test(args.time_test, loader)
        exit(0)
    
    out_tensors = next(iter(loader))
    import imageio
    writer = imageio.get_writer('test_frames.gif')
    for t in range(out_tensors[0].shape[1]):
        writer.append_data(np.concatenate([b for b in out_tensors[0][:, t, 0]], axis=-2))
    writer.close()
    import pdb; pdb.set_trace()
    print('loaded tensors!')
