from ray.tune import Trainable
from robonet.datasets.robonet_dataset import RoboNetDataset
from robonet.datasets import load_metadata
import copy
import torch
import numpy as np
from robonet.awr.models.img_encoder import Encoder
from robonet.awr.models.policy import Policy
from robonet.awr.models.value_fn import ValueFunction


mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).reshape((1, 3, 1, 1)).type(torch.float32)
std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).reshape((1, 3, 1, 1)).type(torch.float32)
class AWRTrainable(Trainable):
    def _setup(self, config):
        self._config = config
        self._lr = config.get('lr', 0.001)
        self._beta = config.get('beta_a', 1)

        hparams = self._default_hparams()
        hparams.update(config.get('data_hparams', {}))
        hparams['load_reward'] = True
        hparams['load_T'] = 2
        self._batch_size = config['batch_size']

        meta_data = load_metadata(config['data_directory'])
        meta_data = meta_data[meta_data['robot'] == 'sawyer']
        meta_data = meta_data[meta_data['adim'] == hparams['target_adim']]
        meta_data = meta_data[meta_data['sdim'] == hparams['target_sdim']]
        sources = [meta_data[meta_data['gripper'] == g] for g in ['wsg50-default_fingers', 'wsg50-grey_fingers', 'wsg50-orange_fingers']]
        sources = [s for s in sources if len(s)]

        self._train_dataset = RoboNetDataset(sources, hparams=hparams)
        self._train_loader = self._train_dataset.make_dataloader(self._batch_size, n_workers=config.get('n_workers', -1))
        self._n_train_steps = config.get('n_steps', 10)
        self._iterator = None
        
        self._v = ValueFunction().cuda()
        self._pi = Policy().cuda()
        self._enc = Encoder().cuda()
        params = list(self._v.parameters()) + list(self._pi.parameters()) + list(self._enc.parameters())
        self._optim = torch.optim.Adam(params, self._lr)
        
    def _default_hparams(self):
        default_dict = {
            'RNG': 0,
            'target_adim': 4,
            'target_sdim': 5,
            'img_size': [240, 320],
            'splits':[0.9, 0.05, 0.05],
            'reward_discount': 0.7
        }
        return default_dict

    def _train(self):
        avg_loss = 0    
        for _ in range(self._n_train_steps):
            images, actions, state, rewards = self._get_batch()

            self._optim.zero_grad()
            encoded = self._enc(images, state)
            pred_values = self._v(encoded)
            pred_mean, pred_std = self._pi(encoded)

            l_value = torch.sum(torch.pow(pred_values - rewards, 2))
            # l_actions = 
            loss = l_value

            avg_loss += loss.item()
            loss.backward()
            self._optim.step()
            
        return {'loss': avg_loss / self._n_train_steps}

    def _save(self, checkpoint_dir):
        raise NotImplementedError

    def _restore(self, checkpoints):
        raise NotImplementedError

    @property
    def iteration(self):
        return self._iteration * self._n_train_steps

    # def _log_result(self, result):
    #     raise NotImplementedError

    def _get_batch(self):
        if self._iterator is None:
            self._iterator = iter(self._train_loader)
        
        try:
            images, actions, state, rewards = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._train_loader)
            images, actions, state, rewards = next(self._iterator)
        
        state = state[:, 0]
        actions = actions[:, 0]
        images = images[:, 0, 0]
        images = (images.type(torch.float32) / 255 - mean) / std

        return [r.to('cuda') for r in [images, actions, state, rewards]]
