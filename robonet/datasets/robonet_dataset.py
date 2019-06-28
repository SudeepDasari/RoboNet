from robonet.datasets.base_dataset import BaseVideoDataset
from robonet.datasets.util.hdf5_loader import load_data, default_loader_hparams
from tensorflow.contrib.training import HParams
import numpy as np
import tensorflow as tf
import copy
import multiprocessing


def _load_data(inputs):
    if len(inputs) == 3:
        f_name, file_metadata, hparams = inputs
        return load_data(f_name, file_metadata, hparams)
    elif len(inputs) == 4:
        f_name, file_metadata, hparams, rng = inputs
        return load_data(f_name, file_metadata, hparams, rng)
    raise ValueError


class RoboNetDataset(BaseVideoDataset):
    def _init_dataset(self):
        if self._hparams.load_random_cam and self._hparams.same_cam_across_sub_batch:
            for s in self._metadata:
                min_ncam = min(s['ncam'])
                if sum(s['ncam'] != min_ncam):
                    print('sub-batch has data with different ncam but each same_cam_across_sub_batch=True! Could result in un-even cam sampling')
                    break
        
        assert self._batch_size % self._hparams.sub_batch_size == 0, "sub batches should evenly divide batch_size!"
        assert np.isclose(sum(self._hparams.splits), 1) and all([0 <= i <= 1 for i in self._hparams.splits]), "splits is invalid"
        assert self._hparams.load_T >=0, "load_T should be non-negative!"

        n_train_ex = 0
        self._train_sources, self._val_sources, self._test_sources = [[] for _ in range(3)]
        
        # smallest max step length of all dataset sources 
        min_steps = min([min(min(m.frame['img_T']), min(m.frame['state_T'])) for m in self._metadata])
        if not self._hparams.load_T:
            self._hparams.load_T = min_steps
        else:
            assert self._hparams.load_T <= min_steps, "ask to load {} steps but some records only have {}!".format(self._hparams.min_T, min_steps)

        for metadata in self._metadata:
            train_files, val_files, test_files = self._split_files(metadata)
            if self._hparams.splits[0]:
                assert len(train_files) > 0, "train files requested but non-given"
                self.rng.shuffle(train_files) 
                self._train_sources.append(train_files)
            if self._hparams.splits[1]:
                assert len(val_files) > 0, "val files requested but non-given"
                self.rng.shuffle(val_files) 
                self._val_sources.append(val_files)
            if self._hparams.splits[2]:
                assert len(test_files) > 0, "test files requested but non-given"
                self.rng.shuffle(test_files) 
                self._test_sources.append(test_files)

        output_format = [tf.uint8, tf.float32, tf.float32]        
        if self._hparams.load_annotations:
            output_format = output_format + [tf.float32]
        
        if self._hparams.ret_fnames:
            output_format = output_format + [tf.string]
        output_format = tuple(output_format)

        self._data_loaders = {}
        # DON'T WRITE THIS SECTION WITH A FOR LOOP
        # YOU MUST CALL next(generator) OR ELSE TENSORFLOW SOMETIMES CRASHES
        # IF YOU KNOW WHY THESE BUGS HAPPENS YOU DESERVE A TURING AWARD
        # TRAIN
        n_workers = min(self._batch_size, multiprocessing.cpu_count())
        if self._hparams.pool_workers:
            n_workers = min(self._hparams.pool_workers, multiprocessing.cpu_count())
        self._pool = multiprocessing.Pool(n_workers)

        if len(self._train_sources):
            n_train_ex = sum(len(f) for f in self._train_sources)
            train_generator = self._hdf5_generator(self._train_sources, self.train_rng, 'train')
            next(train_generator)
            dataset = tf.data.Dataset.from_generator(lambda: train_generator, output_format)
            dataset = dataset.map(self._get_dict).prefetch(self._hparams.buffer_size)
            self._data_loaders['train'] = dataset.make_one_shot_iterator().get_next()
        else:
            print('no train files')

        if len(self._val_sources):
            val_generator = self._hdf5_generator(self._val_sources, self.val_rng, 'val')
            next(val_generator)
            dataset = tf.data.Dataset.from_generator(lambda: val_generator, output_format)
            dataset = dataset.map(self._get_dict).prefetch(max(int(self._hparams.buffer_size / 10), 1))
            self._data_loaders['val'] = dataset.make_one_shot_iterator().get_next()
        else:
            print('no val files')
        
        if len(self._test_sources):
            test_generator = self._hdf5_generator(self._test_sources, self.test_rng, 'test')
            next(test_generator)
            dataset = tf.data.Dataset.from_generator(lambda: test_generator, output_format)
            dataset = dataset.map(self._get_dict).prefetch(max(int(self._hparams.buffer_size / 10), 1))
            self._data_loaders['test'] = dataset.make_one_shot_iterator().get_next()
        else:
            print('no test files')

        return n_train_ex

    def _split_files(self, metadata):
        files = metadata.files
        train_files, val_files, test_files = None, None, None
        splits = np.cumsum([int(i * len(files)) for i in self._hparams.splits]).tolist()
       
        # give extra fat to val set
        if splits[-1] < len(files):
            diff = len(files) - splits[-1]
            for i in range(1, len(splits)):
                splits[i] += diff
        
        if self._hparams.splits[0]:
            train_files = files[:splits[0]]
        if self._hparams.splits[1]:
            val_files = files[splits[0]: splits[1]]
        if self._hparams.splits[2]:
            test_files = files[splits[1]: splits[2]]
        
        return train_files, val_files, test_files

    def _get(self, key, mode):
        return self._data_loaders[mode][key]
    
    def __contains__(self, item):
        return any([item in self._data_loaders.get(m, {}) for m in self.modes])

    @staticmethod
    def _get_default_hparams():
        default_dict = {
            'RNG': 11381294392481135266,             # random seed to initialize data loader rng
            'use_random_train_seed': False,          # use a random seed for training objects 
            'sub_batch_size': 1,                     # sub batch to sample from each source
            'splits': (0.9, 0.05, 0.05),             # train, val, test
            'num_epochs': None,                      # maximum number of epochs (None if iterate forever)
            'ret_fnames': False,                     # return file names of loaded hdf5 record
            'buffer_size': 10,                        # examples to prefetch
            'all_modes_max_workers': True,           # use multi-threaded workers regardless of the mode
            'load_random_cam': True,                 # load a random camera for each example
            'same_cam_across_sub_batch': False,      # same camera across sub_batches
            'pool_workers': 0                        # number of workers for pool (if 0 uses batch_size workers)
        }
        for k, v in default_loader_hparams().items():
            default_dict[k] = v
        
        return HParams(**default_dict)

    def _hdf5_generator(self, sources, rng, mode): 
        file_indices, source_epochs = [[0 for _ in range(len(sources))] for _ in range(2)]

        while True:
            file_hparams = [copy.deepcopy(self._hparams) for _ in range(self._batch_size)]
            if self._hparams.RNG:
                file_rng = [rng.getrandbits(64) for _ in range(self._batch_size)]
            else:
                file_rng = [None for _ in range(self._batch_size)]
            
            file_names, file_metadata = [], []
            b = 0
            while len(file_names) < self._batch_size:
                # if # sources <= # sub_batches then sample each source at least once per batch
                if len(sources) <= self._batch_size // self._hparams.sub_batch_size and b // self._hparams.sub_batch_size < len(sources):
                    selected_source = b // self._hparams.sub_batch_size
                else:
                    selected_source = rng.randint(0, len(sources) - 1)

                for sb in range(self._hparams.sub_batch_size):
                    selected_file = sources[selected_source][file_indices[selected_source]]
                    file_indices[selected_source] += 1
                    selected_file_metadata = self._metadata[selected_source].get_file_metadata(selected_file)

                    file_names.append(selected_file)
                    file_metadata.append(selected_file_metadata)
                    
                    if file_indices[selected_source] >= len(sources[selected_source]):
                        file_indices[selected_source] = 0
                        source_epochs[selected_source] += 1

                        if mode == 'train' and self._hparams.num_epochs is not None and source_epochs[selected_source] >= self._hparams.num_epochs:
                            break
                        rng.shuffle(sources[selected_source])

                b += self._hparams.sub_batch_size

            if self._hparams.load_random_cam:
                b = 0
                while b < self._batch_size:
                    if self._hparams.same_cam_across_sub_batch:
                        selected_cam = [rng.randint(0, min([file_metadata[b + sb]['ncam'] for sb in range(self._hparams.sub_batch_size)]) - 1)]
                        for sb in range(self._hparams.sub_batch_size):
                            file_hparams[b + sb].cams_to_load = selected_cam
                        b += self._hparams.sub_batch_size
                    else:
                        file_hparams[b].cams_to_load = [rng.randint(0, file_metadata[b]['ncam'] - 1)]
                        b += 1

            batch_jobs = [(fn, fm, fh, fr) for fn, fm, fh, fr in zip(file_names, file_metadata, file_hparams, file_rng)]
            batches = self._pool.map_async(_load_data, batch_jobs).get()

            ret_vals = []
            for i, b in enumerate(batches):
                if i == 0:
                    for value in b:
                        ret_vals.append([value[None]])
                else:
                    for v_i, value in enumerate(b):
                        ret_vals[v_i].append(value[None])

            ret_vals = [np.concatenate(v) for v in ret_vals]
            if self._hparams.ret_fnames:
                ret_vals = ret_vals + [file_names]

            yield tuple(ret_vals)

    def _get_dict(self, *args):
        if self._hparams.ret_fnames and self._hparams.load_annotations:
            images, actions, states, annotations, f_names = args
        elif self._hparams.ret_fnames:
            images, actions, states, f_names = args
        elif self._hparams.load_annotations:
            images, actions, states, annotations = args
        else:
            images, actions, states = args
        
        out_dict = {}
        height, width = self._hparams.img_size
        
        if self._hparams.load_random_cam:
            ncam = 1
        else:
            ncam = len(self._hparams.cams_to_load)
        
        shaped_images = tf.reshape(images, [self.batch_size, self._hparams.load_T, ncam, height, width, 3])
        out_dict['images'] = tf.cast(shaped_images, tf.float32) / 255.0
        out_dict['actions'] = tf.reshape(actions, [self.batch_size, self._hparams.load_T - 1, self._hparams.target_adim])
        out_dict['states'] = tf.reshape(states, [self.batch_size, self._hparams.load_T, self._hparams.target_sdim])

        if self._hparams.load_annotations:
            out_dict['annotations'] = tf.reshape(annotations, [self._batch_size, self._hparams.load_T, ncam, height, width, 2])
        if self._hparams.ret_fnames:
            out_dict['f_names'] = f_names
        
        return out_dict


def _timing_test(N, loader):
    import time
    import random

    mode_tensors = {}
    for m in ['train', 'test', 'val']:
        mode_tensors[m] = [loader[x, m] for x in ['images', 'states', 'actions']]
    s = tf.Session()

    timings = []
    for i in range(N):
        m = random.choice(['train', 'test', 'val'])
        start = time.time()
        s.run(mode_tensors[m])
        run_time = time.time() - start
        if m == 'train':
            timings.append(run_time)
        print('run {}, mode {} took {} seconds'.format(i, m, run_time))

    if timings:
        print('train runs took on average {} seconds'.format(sum(timings) / len(timings)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="calculates or loads meta_data frame")
    parser.add_argument('path', help='path to files containing hdf5 dataset')
    parser.add_argument('--robots', type=str, nargs='+', default=None, help='will construct a dataset with batches split across given robots')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for test loader (should be even for non-time test demo to work)')
    parser.add_argument('--mode', type=str, default='train', help='mode to grab data from')
    parser.add_argument('--time_test', type=int, default=0, help='if value provided will run N timing tests')
    parser.add_argument('--load_steps', type=int, default=0, help='if value is provided will load <load_steps> steps')
    args = parser.parse_args()

    hparams = {'RNG': 0, 'ret_fnames': True, 'load_T': args.load_steps, 'sub_batch_size': 2, 'action_mismatch': 3, 'state_mismatch': 3, 'splits':[0.8, 0.1, 0.1]}
    if args.robots:
        from robonet.datasets import load_metadata
        meta_data = load_metadata(args.path)
        hparams['same_cam_across_sub_batch'] = True
        loader = RoboNetDataset(args.batch_size, [meta_data[meta_data['robot'] == r] for r in args.robots], hparams=hparams)
    else:
        loader = RoboNetDataset(args.batch_size, args.path, hparams=hparams)
    
    if args.time_test:
        _timing_test(args.time_test, loader)
        exit(0)

    tensors = [loader[x, args.mode] for x in ['images', 'states', 'actions', 'f_names']]
    s = tf.Session()
    out_tensors = s.run(tensors)
    
    import imageio
    writer = imageio.get_writer('test_frames.gif')
    for t in range(out_tensors[0].shape[1]):
        writer.append_data((np.concatenate([b for b in out_tensors[0][:, t, 0]], axis=-2) * 255).astype(np.uint8))
    writer.close()
    import pdb; pdb.set_trace()
    print('loaded tensors!')