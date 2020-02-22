from robonet.datasets.util.hdf5_loader import HDF5Loader
from robonet_dataset import RoboNetDataset
import numpy as np
import random
import torch


class GraspLabeledDataset(RoboNetDataset):
    def _check_params(self, sources):
        return

    @staticmethod
    def _get_default_hparams():
        default_dict = {
            'splits': (0.9, 0.05, 0.05),             # percentage of files in train, val, test per source
            'train_ex_per_source': None,             # list of train_examples per source (set to None to rely on splits only)
            'normalize_images': True,                # normalize image pixels by torchvision std/mean values
            'reward_discount': 0.9,                  # discount term to be applied while calculating trajectory reward
            'camera': None                           # camera to load images from (randomly choose if None)
        }
        default_dict.update(HDF5Loader.default_hparams())
        return default_dict
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_name, metadata = self._data[idx]
        loader = HDF5Loader(file_name, metadata, self._hparams)

        # get action/state vectors and finger sensors
        states, actions = loader.load_states(), loader.load_actions()
        finger_sensors = loader.hf['env']['finger_sensors'][:].reshape((-1, 1))
        
        if len(finger_sensors.shape) == 1:
            finger_sensors = finger_sensors[:-1]
        else:
            assert len(finger_sensors.shape) == 2, "shape should be 2 or 1!"
            finger_sensors = finger_sensors[:-1, 0]
        
        # calculate grasp label for each time-step
        good_states = np.logical_and(states[1:, 2] >= 0.9, states[1:, -1] > 0)
        good_states = np.logical_and(finger_sensors > 0, good_states).astype(np.float32)
        reward_table = good_states - (1 - good_states) * 0.02

        # get time of grasp and sample s_t *before* that point
        goal_T = np.argmax(good_states)
        s_time = random.randint(0, goal_T - 1)

        # calculate discounted reward        
        reward_slice = reward_table[s_time:]
        discount = np.power(self._hparams['reward_discount'], np.arange(reward_slice.shape[0]))
        reward = np.sum(discount * reward_slice)

        # get images for state/goal
        if self._hparams['camera'] is None:
            cam = random.randint(0, metadata['ncam'] - 1)
        else:
            cam = self._hparams['camera']
        
        state_images = self._proc_images(loader.load_video(cam, start_time=s_time, n_load=2))
        goal_image = self._proc_images(loader.load_video(cam, start_time=goal_T, n_load=1))

        s_t = {'image': state_images[0], 'robot-state': states[s_time]}
        s_t_1 = {'image': state_images[1], 'robot-state': states[s_time + 1]}
        goal_state = {'image': goal_image[0], 'robot-state': states[goal_T]}

        # return (state, action, next_state, reward, goal)
        return s_t, actions[s_time], s_t_1, reward, goal_state


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="calculates or loads meta_data frame")
    parser.add_argument('path', help='path to files containing hdf5 dataset')
    args = parser.parse_args()

    from robonet.datasets import load_metadata
    from torch.utils.data import DataLoader

    metadata = load_metadata(args.path)
    metadata = metadata[metadata['object_grasped'] == True]

    hparams = {'action_mismatch': 3, 'state_mismatch': 3, 'splits':[0.8, 0.1, 0.1], 'normalize_images':False}
    dataset = GraspLabeledDataset(metadata, 'train', hparams)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    for s_t, actions, s_t_1, reward, goal_state in loader:
        import pdb; pdb.set_trace()
