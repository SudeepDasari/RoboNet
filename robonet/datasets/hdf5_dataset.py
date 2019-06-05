from robonet.datasets.base_dataset import BaseVideoDataset
import h5py
import tensorflow as tf
import numpy as np
import cv2
import glob
import random
import math
import imageio
import io
from collections import OrderedDict
import copy
import multiprocessing


def _load_batch(assignment):
    filename, img_T, ncam, img_dims, is_mp4, load_single_rand_cam = assignment
    height, width = img_dims

    if load_single_rand_cam:
        ncam = 1
        rand_cam = random.randint(0, ncam - 1)
        cam_range = range(rand_cam, rand_cam + 1)
    else:
        cam_range = range(ncam)
    
    with h5py.File(filename, 'r') as hf:
        images = np.zeros((ncam, img_T, height, width, 3), dtype=np.uint8)
        if is_mp4:
            for n in cam_range:
                buf = io.BytesIO(hf['env']['cam{}_video'.format(n)]['frames'][:].tostring())
                reader = imageio.get_reader(buf, format='mp4')
                
                for t, img in enumerate(reader):
                    method = cv2.INTER_CUBIC
                    if height * width < img.shape[0] * img.shape[1]:
                        method = cv2.INTER_AREA
                    images[n, t] = cv2.resize(img[:,:,::-1], (width, height), interpolation=method)
            
        else:
            for n in cam_range:
                for t in range(img_T):
                    img = cv2.imdecode(hf['env']['cam{}_video'.format(n)]['frame{}'.format(t)][:], cv2.IMREAD_COLOR)

                    method = cv2.INTER_CUBIC
                    if height * width < img.shape[0] * img.shape[1]:
                        method = cv2.INTER_AREA
                    images[n, t] = cv2.resize(img, (width, height), interpolation=method)
        images = np.swapaxes(images, 0, 1)

        actions = hf['policy']['actions'][:].astype(np.float32)
        states = hf['env']['state'][:].astype(np.float32)

    return actions, images[:,:,:,:,::-1], states


def _slice_helper(tensor, start_i, N, axis):
    shape = tensor.get_shape().as_list()
    assert 0 <= axis < len(shape), "bad axis!"

    starts = [0 for _ in shape]
    ends = shape
    starts[axis], ends[axis] = start_i, N

    return tf.slice(tensor, starts, ends)


def _grab_cam(image_tensor, selected_cams):
    b_dim = image_tensor.get_shape().as_list()[0]
    tensors = []
    for b in range(b_dim):
        tensors.append(image_tensor[b, :, selected_cams[b]][None])
    return tf.concat(tensors, axis=0)


class HDF5VideoDataset(BaseVideoDataset):
    def _init_dataset(self):
        # read hdf5 from base dir and check contents
        dataset_contents = self._get_filenames()
        print('loading {} examples'.format(len(dataset_contents)))
        assert len(dataset_contents), "No hdf5 files in dataset!"

        # consistent shuffling of dataset contents according to set RNG
        dataset_contents.sort()
        self._rand = random.Random(self._hparams.RNG)
        self._rand.shuffle(dataset_contents)

        rand_file = dataset_contents[0]
        with h5py.File(rand_file, 'r') as f:
            assert all([x in f for x in ['env', 'metadata', 'policy', 'misc']])

            self._action_T, self._adim = f['policy']['actions'].shape

            self._ncam = f['env'].attrs.get('n_cams', 0)
            assert self._ncam > 0, "must be at least 1 camera!"

            if 'frames' in f['env']['cam0_video']:
                self._is_mp4 = True
                self._img_T = min([f['env']['cam{}_video'.format(i)]['frames'].attrs['T'] for i in range(self._ncam)])
                self._img_dim = f['env']['cam0_video']['frames'].attrs['shape'][:2]
            else:
                self._is_mp4 = False
                self._img_T = min([len(f['env']['cam{}_video'.format(i)]) for i in range(self._ncam)])
                self._img_dim = f['env']['cam0_video']['frame0'].attrs['shape'][:2]

            self._state_T, self._sdim = f['env']['state'].shape
            assert self._state_T == self._img_T, "#images should match #states!"

            self._valid_keys = ['actions', 'images', 'state']
            self._parser_dtypes = [tf.float32, tf.string, tf.float32]
        
        self._mode_datasets = self._init_queues(dataset_contents)
        self._rand_start = None
        if not self._hparams.load_single_rand_cam:
            self._rand_cam = None
    
    def _gen_hdf5(self, files, mode):
        p = multiprocessing.Pool(self._batch_size)
        files = copy.deepcopy(files)
        i, n_epochs = 0, 0

        while True:
            if i + self._batch_size > len(files):
                i, n_epochs = 0, n_epochs + 1
                if mode == 'train' and self._hparams.num_epochs is not None and n_epochs >= self._hparams.num_epochs:
                    break
                self._rand.shuffle(files)
            batch_files = files[i:i+self._batch_size]
            batch_files = [(f, self._img_T, self._ncam, self._hparams.img_dims, self._is_mp4, self._hparams.load_single_rand_cam) for f in batch_files]
            i += self._batch_size
            batches = p.map(_load_batch, batch_files)
            actions, images, states = [], [], []
            for b in batches:
                for value, arr in zip(b, [actions, images, states]):
                    arr.append(value[None])
            actions, images, states = [np.concatenate(arr, axis=0) for arr in [actions, images, states]]
            yield (actions, images, states)
    
    def _get_dict_act_img_state(self, actions, images, states):
        out_dict = self._get_dict_act_img(actions, images)
        states = tf.reshape(states, [self._batch_size, self._state_T, self._sdim])
        out_dict['states'] = states
        return out_dict
    
    def _get_dict_act_img(self, actions, images):
        actions = tf.reshape(actions, [self._batch_size, self._action_T, self._adim])
        load_cam = self._ncam
        if self._hparams.load_single_rand_cam:
            load_cam = 1
        images = tf.reshape(images, [self._batch_size, self._img_T, load_cam, self._hparams.img_dims[0], self._hparams.img_dims[1], 3])
        return {'actions': actions, 'images': tf.cast(images, tf.float32)}
    
    def _init_queues(self, hdf5_files):
        assert len(self.MODES) == len(self._hparams.splits), "len(splits) should be the same as number of MODES!"
        split_lengths = [int(math.ceil(len(hdf5_files) * x)) for x in self._hparams.splits[1:]]
        split_lengths = np.cumsum([0, len(hdf5_files) - sum(split_lengths)] + split_lengths)
        splits = OrderedDict()
        for i, name in enumerate(self.MODES):
            splits[name] = hdf5_files[split_lengths[i]:split_lengths[i+1]]
        self._num_ex = len(splits['train'])

        mode_datasets = {}
        for name, files in splits.items():
            assert 'state' in self._valid_keys, "assume all records have state"
            dataset = tf.data.Dataset.from_generator(lambda:self._gen_hdf5(files, name), (tf.float32, tf.uint8, tf.float32))
            dataset = dataset.map(self._get_dict_act_img_state).prefetch(100)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()

            output_element = {}
            for k in list(next_element.keys()):
                output_element[k] = tf.reshape(next_element[k],
                                               [self._batch_size] + next_element[k].get_shape().as_list()[1:])
            
            mode_datasets[name] = output_element
        return mode_datasets

    def _get(self, key, mode):
        assert key in self._mode_datasets[mode], "Key {} is not recognized for mode {}".format(key, mode)

        return self._mode_datasets[mode][key]
    
    @staticmethod
    def _get_default_hparams():
        default_params = super(HDF5VideoDataset, HDF5VideoDataset)._get_default_hparams()
        
        # set None if you want a random seed for dataset shuffling
        default_params.add_hparam('RNG', 11381294392481135266)
        default_params.add_hparam('splits', [0.9, 0.05, 0.05])       # (train, val, test) split
        default_params.add_hparam('img_dims', (48, 64))
        default_params.add_hparam('max_start', -1)
        default_params.add_hparam('load_single_rand_cam', True)      # if True then only load one camera at random instead of all frames
        
        return default_params

    @property
    def ncam(self):
        return self._ncam

    def make_input_targets(self, n_frames, n_context, mode, img_dtype=tf.float32):
        assert n_frames > n_context and n_context >= 0
        if self._rand_start is None:
            img_T = self._get('images', mode).get_shape().as_list()[1]
            self._rand_start = tf.random_uniform((), maxval=img_T - n_frames + 1, dtype=tf.int32)

        if not self._hparams.load_single_rand_cam and self._rand_cam is None:
            n_cam =  self._get('images', mode).get_shape().as_list()[2]
            self._rand_cam = tf.random_uniform((self._batch_size,), maxval=n_cam, dtype=tf.int32)
        
        inputs = OrderedDict()
        img_slice =  _slice_helper(self._get('images', mode), self._rand_start, n_frames, 1) 
        targets_slice = _slice_helper(self._get('images', mode), self._rand_start + n_context, n_frames - n_context, 1)
        if not self._hparams.load_single_rand_cam:
            img_slice = _grab_cam(img_slice, self._rand_cam)
            targets = _grab_cam(targets_slice, self._rand_cam)
        else:
            targets = targets_slice[:, :, 0]
            img_slice = img_slice[:, :, 0]
        
        inputs['images'] = tf.cast(img_slice / 255.0, img_dtype)
        inputs['states'] = _slice_helper(self._get('states', mode), self._rand_start, n_frames, 1)
        inputs['actions'] = _slice_helper(self._get('actions', mode), self._rand_start, n_frames-1, 1)
        
        targets = tf.cast(targets / 255.0, img_dtype)
        return inputs, targets
    
    @property
    def num_examples_per_epoch(self):
        return self._num_ex


if __name__ == '__main__':
    import moviepy.editor as mpy
    import argparse
    import time

    parser = argparse.ArgumentParser(description="converts dataset from pkl format to hdf5")
    parser.add_argument('input_folder', type=str, help='folder containing hdf5 files')
    parser.add_argument('--N', type=int, help='number of timings', default=20)
    args = parser.parse_args()

    path = args.input_folder
    batch_size = 16
    # small shuffle buffer for testing
    dataset = HDF5VideoDataset(path, batch_size, hparams_dict={'buffer_size':10})

    images, actions = dataset['images'], dataset['actions']
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    timings = []
    for i in range(args.N):
        start = time.time()
        imgs, acts = sess.run([images[0], actions[0]])
        timings.append(time.time()- start)
        print('batch {} took: {}'.format(i, timings[-1]))
    print('avg time', sum(timings) / len(timings))
    
    for i in range(imgs.shape[1]):
        mpy.ImageSequenceClip([fr for fr in imgs[:, i]], fps=5).write_gif('test{}.gif'.format(i))
    print(acts)
