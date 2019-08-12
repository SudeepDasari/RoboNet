import os
import tensorflow as tf
from tensorflow.contrib.training import HParams
import glob
from robonet.datasets.base_dataset import BaseVideoDataset
import random
import functools
import json


class TPUVideoDataset(BaseVideoDataset):
    def __init__(self, dataset_batches, dataset_paths, hparams=dict()):
        self._hparams = self._get_default_hparams().override_from_dict(hparams)                 # initialize hparams and store metadata_frame
        self._init_rng()                                                                        # init rng objects
        
        assert isinstance(dataset_batches, (list, tuple)), "dataset_batches must be a list of batch_sizes per source"
        assert isinstance(dataset_paths, (list, tuple)), "dataset_batches must be a list of paths per source"
        self._batch_size = sum(dataset_batches)

        self._source_batch_sizes = dataset_batches
        self._source_dataset_paths = dataset_paths

        self._init_dataset()

    def _init_dataset(self):
        self._mode_datasets = {}
        for m in self.modes:
            self._mode_datasets[m] = []

        for batch_size, dataset_path in zip(self._source_batch_sizes, self._source_dataset_paths):
            assert batch_size > 0
            assert 0 < self._hparams.train_frac < 1
            assert self._hparams.load_T > 1

            dataset_metadata = json.load(open('{}/format.json'.format(dataset_path), 'r'))
            
            if self._hparams.bucket_dir:
                print('loading files from: {}'.format(dataset_path + '/files.json'))
                all_files = json.load(open(dataset_path + '/files.json'))
                all_files = ['{}/{}'.format(self._hparams.bucket_dir, f) for f in all_files]
            else:
                all_files = glob.glob('{}/*.tfrecord'.format(dataset_path))
            all_files.sort(key=lambda x: x.split('/')[-1])
            
            self._random_generator['base'].shuffle(all_files)
            pivot = max(int(len(all_files) * self._hparams.train_frac), 1)
            train_f, val_f = all_files[:pivot], all_files[pivot:]

            self._random_generator['val'].shuffle(val_f)
            self._random_generator['train'].shuffle(train_f)

            for m, files in zip(self.modes, [train_f, val_f]):
                outputs = self._build_dataset(files, m, dataset_metadata, batch_size)

                # enforces static shapes constraint
                height, width = dataset_metadata['img_dim']
                outputs['images'] = tf.cast(tf.reshape(outputs['images'], [batch_size, self._hparams.load_T, height, width, 3]), tf.float32) / 255
                outputs['actions'] = tf.reshape(outputs['actions'], [batch_size, self._hparams.load_T - 1, dataset_metadata['adim']])
                outputs['states'] = tf.reshape(outputs['states'], [batch_size, self._hparams.load_T, dataset_metadata['adim']])

                self._mode_datasets[m].append(outputs)

        for m in self.modes:
            tensor_list = self._mode_datasets.pop(m)
            self._mode_datasets[m] = {}
            for key in ['images', 'states', 'actions']:
                self._mode_datasets[m][key] = tf.concat([out_dict[key] for out_dict in tensor_list], axis=0)
    
    def _build_dataset(self, files, mode, dataset_metadata, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        if mode == 'train':
            dataset = dataset.repeat(self._hparams.n_epochs)
        else:
            dataset = dataset.repeat(None)    # always have infinite val records

        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        dataset = dataset.with_options(ignore_order)
        dataset = dataset.interleave(tf.data.TFRecordDataset, 
                                    cycle_length=min(len(files), 32),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

        parse_fn = functools.partial(self._parse_records, metadata=dataset_metadata)
        dataset = dataset.map(parse_fn)
        dataset = dataset.shuffle(buffer_size=self._hparams.shuffle_buffer)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        outputs = dataset.make_one_shot_iterator().get_next()
        return outputs

    def _parse_records(self, serialized_example, metadata):
        feat_names = {}
        feat_names['images'] = tf.FixedLenFeature([], tf.string)
        feat_names['actions'] = tf.FixedLenFeature([(metadata['T'] - 1) * metadata['adim']], tf.float32)
        feat_names['states'] = tf.FixedLenFeature([metadata['T'] * metadata['sdim']], tf.float32)

        feature = tf.parse_single_example(serialized_example, features=feat_names)

        rand_start = tf.random.uniform((), 0, metadata['T'] - self._hparams.load_T, dtype=tf.int32)
        rand_cam = tf.random.uniform((), 0, metadata['ncam'], dtype=tf.int32)

        decoded_feat = {}
        height, width = metadata['img_dim']
        
        vid_decode = tf.reshape(tf.image.decode_jpeg(feature['images'], channels=3), (metadata['T'] * metadata['ncam'] * height, width, 3))
        decoded_feat['images'] = tf.reshape(vid_decode, [metadata['T'], metadata['ncam'], height, width, 3])[rand_start:rand_start+self._hparams.load_T, rand_cam]
        decoded_feat['actions'] = tf.reshape(feature['actions'], [metadata['T'] - 1, metadata['adim']])[rand_start:rand_start+self._hparams.load_T - 1]
        decoded_feat['states'] = tf.reshape(feature['states'], [metadata['T'], metadata['sdim']])[rand_start:rand_start+self._hparams.load_T]

        return decoded_feat

    def _get(self, key, mode):
        return self._mode_datasets[mode][key]

    @staticmethod
    def _get_default_hparams():
        default_dict = {
            'RNG': 11381294392481135266,
            'use_random_train_seed': False,
            'shuffle_buffer': 500,
            'n_epochs': None,
            'buffer_size': 10,
            'train_frac': 0.9,                  # train, val
            'load_T': 15,
            'bucket_dir': ''
        }
        return HParams(**default_dict)
    
    def __contains__(self, item):
        return item in ['images', 'actions', 'states']

    @property
    def modes(self):
        return ['train', 'val']

    @property   
    def num_examples_per_epoch(self):
        raise NotImplementedError


if __name__ == '__main__':
    import argparse
    import imageio
    import numpy as np
    import time


    parser = argparse.ArgumentParser(description="tfrecord dataset tester")
    parser.add_argument('--path', type=str, required=True, help='path to tfrecord files')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for loaded data')
    args = parser.parse_args()

    loader = TPUVideoDataset([args.batch_size], [args.path], {'train_frac': 0.5, 'shuffle_buffer': 10})
    print(loader['images'], loader['actions'], loader['states'])
    s = tf.Session()
    for j in range(10):
        t = time.time()
        img, act, state = s.run([loader['images'], loader['actions'], loader['states']])
        print(time.time() - t)
        print('actions', act)
        print('state', state)
    
        w = imageio.get_writer('./out{}.gif'.format(j))
        for t in range(img.shape[1]):
            w.append_data((np.concatenate(img[:, t], axis=-2) * 255).astype(np.uint8))

    import pdb; pdb.set_trace()
    print(img.shape)
