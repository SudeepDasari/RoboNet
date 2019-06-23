from robonet.datasets.base_dataset import BaseVideoDataset
from robonet.datasets.util.hdf5_loader import load_data, default_loader_hparams
from tensorflow.contrib.training import HParams
import numpy as np
import tensorflow as tf
import functools
import multiprocessing


class RoboNetDataset(BaseVideoDataset):
    def _init_dataset(self):
        assert np.isclose(sum(self._hparams.splits), 1) and all([0 <= i <= 1 for i in self._hparams.splits]), "splits is invalid"
        assert self._hparams.load_T >=0, "load_T should be non-negative!"

        self._train_files, self._val_files, self._test_files = [[] for _ in range(3)]
        self._init_train_val_test_files()

        output_format = [tf.uint8, tf.float32, tf.float32]        
        if self._hparams.load_annotations:
            output_format = output_format + [tf.float32]
        
        if self._hparams.ret_fnames:
            output_format = output_format + [tf.string]
        output_format = tuple(output_format)

        self._data_loaders = {}
        # DON'T WRITE THIS SECTION WITH A FOR LOOP DUE TO BUG WITH TENSORFLOW DATASET
        # TRAIN
        if len(self._train_files):
            train_generator = self._hdf5_generator(self._train_files, self.train_rng, 'train')
            dataset = tf.data.Dataset.from_generator(lambda: train_generator, output_format)
            dataset = dataset.map(self._get_dict).prefetch(self._hparams.buffer_size)
            self._data_loaders['train'] = dataset.make_one_shot_iterator().get_next()
        else:
            print('no train files')

        if len(self._val_files):
            val_generator = self._hdf5_generator(self._val_files, self.val_rng, 'val')
            dataset = tf.data.Dataset.from_generator(lambda: val_generator, output_format)
            dataset = dataset.map(self._get_dict).prefetch(max(int(self._hparams.buffer_size / 10), 1))
            self._data_loaders['val'] = dataset.make_one_shot_iterator().get_next()
        else:
            print('no val files')
        
        if len(self._test_files):
            test_generator = self._hdf5_generator(self._test_files, self.test_rng, 'test')
            dataset = tf.data.Dataset.from_generator(lambda: test_generator, output_format)
            dataset = dataset.map(self._get_dict).prefetch(max(int(self._hparams.buffer_size / 10), 1))
            self._data_loaders['test'] = dataset.make_one_shot_iterator().get_next()
        else:
            print('no test files')

        return len(self._train_files)
    
    def _init_train_val_test_files(self):
        files, min_steps = self._metadata.files, int(min(min(self._metadata.frame['img_T']), min(self._metadata.frame['state_T'])))
        if not self._hparams.load_T:
            self._hparams.load_T = min_steps
        else:
            assert self._hparams.load_T <= min_steps, "ask to load {} steps but some records only have {}!".format(self._hparams.min_T, min_steps)
        self.rng.shuffle(files)

        splits = np.cumsum([int(i * len(files)) for i in self._hparams.splits]).tolist()
        # give extra fat to val set
        if splits[-1] < len(files):
            diff = len(files) - splits[-1]
            for i in range(1, len(splits)):
                splits[i] += diff
        
        if self._hparams.splits[0]:
            self._train_files = files[:splits[0]]
        if self._hparams.splits[1]:
            self._val_files = files[splits[0]: splits[1]]
        if self._hparams.splits[2]:
            self._test_files = files[splits[1]: splits[2]]

    def _get(self, key, mode):
        return self._data_loaders[mode][key]
    
    def __contains__(self, item):
        return any([item in self._data_loaders.get(m, {}) for m in self.modes])

    @staticmethod
    def _get_default_hparams():
        default_dict = {
            'RNG': 11381294392481135266,
            'splits': (0.9, 0.05, 0.05),      # train, val, test
            'num_epochs': None,
            'ret_fnames': False,
            'buffer_size': 1000,
            'all_modes_max_workers': True
        }
        for k, v in default_loader_hparams().items():
            default_dict[k] = v
        
        return HParams(**default_dict)

    def _hdf5_generator(self, files, rng, mode):
        n_workers = max(1, int(self._batch_size // 5))
        if self._hparams.all_modes_max_workers or mode == 'train':
            n_workers = self._batch_size
        
        p = multiprocessing.Pool(n_workers)
        file_index, n_epochs = 0, 0

        while True:
            file_names = []
            while len(file_names) < self._batch_size:
                if file_index >= len(files):
                    file_index, n_epochs = 0, n_epochs + 1
                    if mode == 'train' and self._hparams.num_epochs is not None and n_epochs >= self._hparams.num_epochs:
                        break
                    rng.shuffle(files)
                file_names.append(files[file_index])
                file_index += 1
            
            if self._hparams.RNG:
                rng_seeds = [rng.getrandbits(64) for _ in range(self._batch_size)]
            else:
                rng_seeds = [None for _ in range(self._batch_size)]

            map_fn = functools.partial(load_data, hparams=self._hparams)
            batch_files = [(b_seed, f, self._metadata.get_file_metadata(f)) for b_seed, f in zip(rng_seeds, file_names)]

            batches = p.map(map_fn, batch_files)
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


def _timing_test(N, path, batch_size):
    import time
    loader = RoboNetDataset(batch_size, path)
    tensors = [loader[x] for x in ['images', 'states', 'actions']]
    s = tf.Session()

    timings = []
    for i in range(N):
        start = time.time()
        s.run(tensors)
        timings.append(time.time() - start)

        print('run {} took {} seconds'.format(i, timings[-1]))
    print('runs took on average {} seconds'.format(sum(timings) / len(timings)))


if __name__ == '__main__':
    import argparse
    import pdb

    parser = argparse.ArgumentParser(description="calculates or loads meta_data frame")
    parser.add_argument('path', help='path to files containing hdf5 dataset')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for test loader')
    parser.add_argument('--time_test', type=int, default=0, help='if value provided will run N timing tests')
    parser.add_argument('--load_steps', type=int, default=0, help='if value is provided will load <load_steps> steps')
    parser.add_argument('--load_annotations', action='store_true', help='tests annotation loading')
    args = parser.parse_args()
    
    if args.time_test:
        _timing_test(args.time_test, args.path, args.batch_size)
        exit(0)

    hparams = {'ret_fnames': True, 'load_T': args.load_steps}
    if args.load_annotations:
        from robonet.datasets import load_metadata
        meta_data = load_metadata(args.path)
        meta_data = meta_data[meta_data['contains_annotation'] == True]
        hparams['load_annotations'] = True
        hparams['splits'] = [0.8, 0.1, 0.1]

        loader = RoboNetDataset(args.batch_size, metadata_frame=meta_data, hparams=hparams)
        tensors = [loader[x] for x in ['images', 'states', 'actions', 'annotations', 'f_names']]
        tensors = tensors + [loader[x, 'val'] for x in ['images', 'states', 'actions', 'annotations', 'f_names']]
        tensors = tensors + [loader[x, 'test'] for x in ['images', 'states', 'actions', 'annotations', 'f_names']]
    else:
        loader = RoboNetDataset(args.batch_size, args.path, hparams=hparams)
        tensors = [loader[x] for x in ['images', 'states', 'actions', 'f_names']]
    s = tf.Session()
    out_tensors = s.run(tensors)

    pdb.set_trace()
    print('done testing!')