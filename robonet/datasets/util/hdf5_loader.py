import h5py
import cv2
import imageio
import io
import hashlib
import numpy as np
import os
import random
import pickle as pkl


class ACTION_MISMATCH:
    ERROR = 0
    PAD_ZERO = 1
    CLEAVE = 2


class STATE_MISMATCH:
    ERROR = 0
    PAD_ZERO = 1
    CLEAVE = 2


class HDF5Loader:
    def __init__(self, f_name, file_metadata, hparams):
        self._file_metadata = file_metadata
        self._hparams = hparams

        assert os.path.exists(f_name) and os.path.isfile(f_name), "invalid f_name"
        with open(f_name, 'rb') as f:
            buf = f.read()
        assert hashlib.sha256(buf).hexdigest() == file_metadata['sha256'], "file hash doesn't match meta-data. maybe delete meta-data and re-generate?"
        # self._hf = h5py.File(io.BytesIO(buf), 'r')
        self._hf = h5py.File(f_name, 'r')

        # start_time, n_states = 0, min([file_metadata['state_T'], file_metadata['img_T'], file_metadata['action_T'] + 1])
        # assert n_states > 1, "must be more than one state in loaded tensor!"
        # if 1 < hparams['load_T'] < n_states:
        #     start_time = rng.randint(0, n_states - hparams['load_T'])
        #     n_states = hparams['load_T']

        # images = [load_camera_imgs(c, start_time, n_states)[None] for c in hparams['cams_to_load']]
        # images = np.swapaxes(np.concatenate(images, 0), 0, 1)
        # images = np.transpose(images, (0, 1, 4, 2, 3))
        # actions = load_actions(hf, file_metadata, hparams).astype(np.float32)[start_time:start_time + n_states-1]
        # full_state = load_states(hf, file_metadata, hparams).astype(np.float32)
        # states = full_state[start_time:start_time + n_states]

    def load_video(self, cam_index, target_dims=None, start_time=0, n_load=None):
        if target_dims is None:
            target_dims = self._hparams['img_size']
        cam_group = self._hf['env']['cam{}_video'.format(cam_index)]
        old_dims = self._file_metadata['frame_dim']
        length = self._file_metadata['img_T']
        encoding = self._file_metadata['img_encoding']
        image_format = self._file_metadata['image_format']

        if n_load is None and start_time == 0:
            n_load = length
        elif n_load is None:
            raise ValueError("Must supply both start_time and n_load or neither!")

        old_height, old_width = old_dims
        target_height, target_width = target_dims
        resize_method = cv2.INTER_CUBIC
        if target_height * target_width < old_height * old_width:
            resize_method = cv2.INTER_AREA
        
        images = np.zeros((n_load, target_height, target_width, 3), dtype=np.uint8)
        if encoding == 'mp4':
            buf = io.BytesIO(cam_group['frames'][:].tostring())
            img_buffer = [img for t, img in enumerate(imageio.get_reader(buf, format='mp4')) if start_time <= t < n_load + start_time]
        elif encoding == 'jpg':
            img_buffer = [cv2.imdecode(cam_group['frame{}'.format(t)][:], cv2.IMREAD_COLOR)
                                    for t in range(start_time, start_time + n_load)]
        else: 
            raise ValueError("encoding not supported")
        
        for t, img in enumerate(img_buffer):
            if (old_height, old_width) == (target_height, target_width):
                images[t] = img
            else:
                images[t] = cv2.resize(img, (target_width, target_height), interpolation=resize_method)
        if image_format == 'RGB':
            return images
        elif image_format == 'BGR':
            return images[:, :, :, ::-1]
        raise NotImplementedError

    def _add_finger_sensor(self, ret_state):
        if self._hparams['load_finger_sensors']:
            finger_sensors = self._hf['env']['finger_sensors'][:]
            return np.concatenate((ret_state, finger_sensors), -1)
        return ret_state

    def load_states(self):
        s_T, sdim = self._file_metadata ['state_T'], self._file_metadata ['sdim']
        if self._hparams['target_sdim'] == sdim:
            return self._add_finger_sensor(self._hf['env']['state'][:])

        elif sdim < self._hparams['target_sdim'] and self._hparams['state_mismatch'] & STATE_MISMATCH.PAD_ZERO:
            pad = np.zeros((s_T, self._hparams['target_sdim'] - sdim), dtype=np.float32)
            return self._add_finger_sensor(np.concatenate((self._hf['env']['state'][:], pad), axis=-1))

        elif sdim > self._hparams['target_sdim'] and self._hparams['state_mismatch'] & STATE_MISMATCH.CLEAVE:
            return self._add_finger_sensor(self._hf['env']['state'][:][:, :self._hparams['target_sdim']])

        else:
            raise ValueError("file sdim - {}, target sdim - {}, pad behavior - {}".format(sdim, self._hparams['target_sdim'], self._hparams['state_mismatch']))

    def load_actions(self):
        a_T, adim = self._file_metadata['action_T'], self._file_metadata['adim']
        if self._hparams['target_adim'] == adim:
            return self._hf['policy']['actions'][:]

        elif self._hparams['target_adim'] == adim + 1 and self._hparams['impute_autograsp_action'] and self._file_metadata ['primitives'] == 'autograsp':
            action_append, old_actions = np.zeros((a_T, 1)), self._hf['policy']['actions'][:]
            next_state = self._hf['env']['state'][:][1:, -1]
            
            high_val, low_val = self._file_metadata['high_bound'][-1], self._file_metadata['low_bound'][-1]
            midpoint = (high_val + low_val) / 2.0

            for t, s in enumerate(next_state):
                if s > midpoint:
                    action_append[t, 0] = high_val
                else:
                    action_append[t, 0] = low_val
            return np.concatenate((old_actions, action_append), axis=-1)

        elif adim < self._hparams['target_adim'] and self._hparams['action_mismatch'] & ACTION_MISMATCH.PAD_ZERO:
            pad = np.zeros((a_T, self._hparams['target_adim'] - adim), dtype=np.float32)
            return np.concatenate((self._hf['policy']['actions'][:], pad), axis=-1)

        elif adim > self._hparams['target_adim'] and self._hparams['action_mismatch'] & ACTION_MISMATCH.CLEAVE:
            return self._hf['policy']['actions'][:][:, :self._hparams['target_adim']]

        else:
            raise ValueError("file adim - {}, target adim - {}, pad behavior - {}".format(adim, self._hparams['target_adim'], self._hparams['action_mismatch']))

    def load_robot_id(self, robotname_list):
        robotname2id = {n: i for i, n in enumerate(robotname_list)}
        return robotname2id[self._file_metadata['robot']]

    def close(self):
        self._hf.close()
        self._hf = None
    
    @staticmethod
    def default_hparams():
        return {
                'target_adim': 4,
                'target_sdim': 5,
                'state_mismatch': STATE_MISMATCH.ERROR,     # TODO make better flag parsing
                'action_mismatch': ACTION_MISMATCH.ERROR,   # TODO make better flag parsing
                'img_size': [48, 64],
                'impute_autograsp_action': True,
                'load_finger_sensors': False,
                }

    @property
    def hf(self):
        return self._hf

# def load_data(f_name, file_metadata, hparams, rng=None):
#     rng = random.Random(rng)
#     assert os.path.exists(f_name) and os.path.isfile(f_name), "invalid f_name"

#     with open(f_name, 'rb') as f:
#         buf = f.read()
#     assert hashlib.sha256(buf).hexdigest() == file_metadata['sha256'], "file hash doesn't match meta-data. maybe delete pkl and re-generate?"

#     with h5py.File(io.BytesIO(buf), 'r') as hf:
#         start_time, n_states = 0, min([file_metadata['state_T'], file_metadata['img_T'], file_metadata['action_T'] + 1])
#         assert n_states > 1, "must be more than one state in loaded tensor!"
#         if 1 < hparams['load_T'] < n_states:
#             start_time = rng.randint(0, n_states - hparams['load_T'])
#             n_states = hparams['load_T']

#         assert all([0 <= i < file_metadata['ncam'] for i in hparams['cams_to_load']]), "cams_to_load out of bounds!"
#         images = [load_camera_imgs(c, hf, file_metadata, hparams['img_size'], start_time, n_states)[None] for c in hparams['cams_to_load']]
#         images = np.swapaxes(np.concatenate(images, 0), 0, 1)
#         images = np.transpose(images, (0, 1, 4, 2, 3))
#         actions = load_actions(hf, file_metadata, hparams).astype(np.float32)[start_time:start_time + n_states-1]
#         full_state = load_states(hf, file_metadata, hparams).astype(np.float32)
#         states = full_state[start_time:start_time + n_states]

#         if hparams['load_finger_sensors']:
#             finger_sensors = hf['env']['finger_sensors'][:][start_time:start_time + n_states].astype(np.float32).reshape((-1, 1))
#             states = np.concatenate((states, finger_sensors), -1)

#         if hparams['load_reward']:
#             assert 1 >= hparams['reward_discount'] >= 0, 'invalid reward discount'
#             finger_sensors = hf['env']['finger_sensors'][:].reshape((-1, 1))
#             good_states = np.logical_and(full_state[1:, 2] >= 0.9, full_state[1:, -1] > 0)
#             good_states = np.logical_and(finger_sensors[1:, 0] > 0, good_states).astype(np.float32)
#             reward_table = good_states - (1 - good_states) * 0.02
#             rewards = []

#             for s_t in range(start_time, start_time + n_states - 1):
#                 reward_slice = reward_table[s_t:]
#                 discount = np.power(hparams['reward_discount'], np.arange(reward_slice.shape[0]))
#                 rewards.append(np.sum(discount * reward_slice))
#             return images, actions, states, np.array(rewards).astype(np.float32)
    
#         if hparams['load_annotations']:
#             annotations = load_annotations(hf, file_metadata, hparams, hparams['cams_to_load'])[start_time:start_time + n_states]
#             return images, actions, states, annotations

#     return images, actions, states


if __name__ == '__main__':
    import argparse
    from robonet.datasets import load_metadata
    import random
    import matplotlib.pyplot as plt
    import os

    parser = argparse.ArgumentParser(description="tests hdf5 data loader without tensorflow dataset wrapper")
    parser.add_argument('file', type=str, help="path to hdf5 you want to load")
    args = parser.parse_args()
    args.file = os.path.expanduser(args.file)
    
    assert 'hdf5' in args.file
    data_folder = os.path.dirname(args.file)
    meta_data = load_metadata(data_folder)
    hparams = HDF5Loader.default_hparams()

    file_handle = HDF5Loader(args.file, meta_data.get_file_metadata(args.file), hparams)
    import pdb; pdb.set_trace()
    print(file_handle)
