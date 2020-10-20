from torch.utils.data import Dataset
from robonet.datasets.util.hdf5_loader import HDF5Loader
from robonet.datasets.util.dataset_utils import split_train_val_test
import numpy as np
import copy
import torch
import random


class RoboNetDataset(Dataset):
    def __init__(self, metadata_sources, mode='train', hparams=dict()):
        self._hparams = self._get_default_hparams()
        self._hparams.update(hparams)
        
        if isinstance(metadata_sources, (list, tuple)):
            sources = metadata_sources
        else:
            sources = [metadata_sources]
        self._check_params(sources)
        
        self._data = []
        for i, source in enumerate(sources):
            file_shuffle = random.Random(5011757766786901527)
            if self._hparams['train_ex_per_source'] is not None:
                source_files = split_train_val_test(source, train_ex=self._hparams['train_ex_per_source'][i], rng=file_shuffle)[mode]
            else:
                source_files = split_train_val_test(source, splits=self._hparams['splits'], rng=file_shuffle)[mode]
            self._data.extend([(f, source.get_file_metadata(f)) for f in source_files])
        
        # constants for normalization
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, -1, 1, 1))
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, -1, 1, 1))
    
    def _check_params(self, sources):
        assert self._hparams['T'] >= 0, "can't load less than 0 frames!"
        # smallest max step length of all dataset sources 
        min_steps = min([min(min(m.frame['img_T']), min(m.frame['state_T'])) for m in sources])
        if not self._hparams['T']:
            self._hparams['T'] = min_steps
        else:
            assert self._hparams['T'] <= min_steps, "ask to load {} steps but some records only have {}!".format(self._hparams['T'], min_steps)

    @staticmethod
    def _get_default_hparams():
        default_dict = {
            'splits': (0.9, 0.05, 0.05),             # percentage of files in train, val, test per source
            'train_ex_per_source': None,             # list of train_examples per source (set to None to rely on splits only)
            'T': 0,                                  # will load up to T frames if T > 0
            'normalize_images': True                 # normalize image pixels by torchvision std/mean values
        }
        default_dict.update(HDF5Loader.default_hparams())
        return default_dict

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_name, metadata = self._data[idx]
        loader = HDF5Loader(file_name, metadata, self._hparams)
        # get action/state vectors
        states, actions = loader.load_states(), loader.load_actions()

        # get camera and slice states if T is set
        random_camera = random.randint(0, metadata['ncam'] - 1)
        if self._hparams['T']:
            img_len = metadata['img_T']
            start_time = random.randint(0, img_len - self._hparams['T'])

            images = loader.load_video(random_camera, start_time=start_time, n_load=self._hparams['T'])
            states = states[start_time:start_time + self._hparams['T']]
            actions = actions[start_time:start_time + self._hparams['T'] - 1]
        else:
            images = loader.load_video(random_camera)
        images = self._proc_images(images)
        
        return images, states.astype(np.float32), actions.astype(np.float32)
    
    def _proc_images(self, images):
        # cast and normalize images (if set)
        if len(images.shape) == 4:
            images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32) / 255
        if len(images.shape) == 5:
            images = np.transpose(images, (0, 1, 4, 2, 3)).astype(np.float32) / 255
        if self._hparams['normalize_images']:
            images -= self._mean
            images /= self._std
        return images


def _timing_test(N, loader):
    import time

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
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for test loader (should be even for non-time test demo to work)')
    parser.add_argument('--mode', type=str, default='train', help='mode to grab data from')
    parser.add_argument('--time_test', type=int, default=0, help='if value provided will run N timing tests')
    parser.add_argument('--load_steps', type=int, default=0, help='if value is provided will load <load_steps> steps')
    args = parser.parse_args()

    from robonet.datasets import load_metadata
    from torch.utils.data import DataLoader

    metadata = load_metadata(args.path)
    hparams = {'T': args.load_steps, 'action_mismatch': 3, 'state_mismatch': 3, 'splits':[0.8, 0.1, 0.1], 'normalize_images':False}
    dataset = RoboNetDataset(metadata, args.mode, hparams)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    
    if args.time_test:
        _timing_test(args.time_test, loader)
        exit(0)
    
    images, states, actions = next(iter(loader))
    images = np.transpose(images.numpy(), (0, 1, 3, 4, 2))
    images *= 255
    images = images.astype(np.uint8)
    
    print(images.shape)
    import imageio
    writer = imageio.get_writer('test_frames.gif')
    for t in range(images.shape[1]):
        writer.append_data(np.concatenate([b for b in images[:, t]], axis=-2))
    writer.close()

