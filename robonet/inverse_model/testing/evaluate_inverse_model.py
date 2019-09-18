from robonet.video_prediction.testing import VPredEvaluation
from robonet.yaml_util import parse_tune_config as parse_config
import os
import argparse
from robonet.inverse_model.testing.action_inference_interface import ActionInferenceInterface
import tensorflow as tf
from robonet.datasets import get_dataset_class, load_metadata
from tensorflow.contrib.training import HParams
from robonet.datasets.util.tensor_multiplexer import MultiplexedTensors
from robonet.video_prediction.testing.model_evaluation_interface import VPredEvaluation
import cv2
import numpy as np


class DataLoader:
    def __init__(self, config):
        # run hparams are passed in through config dict
        self.dataset_hparams, self.model_hparams, self._hparams = self._extract_hparams(config)
        self._inputs, self._targets = self._make_dataloaders(config)

    def _default_hparams(self):
        default_dict = {
            'batch_size': 16,
            'restore_dir': '',
            'n_gpus': 1,
            'pad_amount': 2,
            'scalar_summary_freq': 100,
            'image_summary_freq': 1000,
            'train_fraction': 0.9,
            'val_fraction': 0.05,
            'max_to_keep': 3,
            'max_steps': 300000,
        }
        return HParams(**default_dict)

    def _get_dataset_class(self, class_name):
        return get_dataset_class(class_name)

    def _extract_hparams(self, config):
        """
        Grabs and (optionally) modifies hparams
        """
        self._batch_config = config.pop('batch_config')
        dataset_hparams, model_hparams = config.pop('loader_hparams', {}), config.pop('model_hparams', {})
        hparams = self._default_hparams().override_from_dict(config)
        hparams.graph_type = model_hparams.pop('graph_type')                      # required key which tells model which graph to load
        assert hparams.max_steps > 0, "max steps must be positive!"

        if 'splits' not in dataset_hparams:
            dataset_hparams['splits'] = [hparams.train_fraction, hparams.val_fraction, 1 - hparams.val_fraction - hparams.train_fraction]
            assert all([x >= 0 for x in dataset_hparams['splits']]), "invalid train/val fractions!"

        if 'sequence_length' in model_hparams and 'load_T' not in dataset_hparams:
            dataset_hparams['load_T'] = model_hparams['sequence_length']
        
        return dataset_hparams, model_hparams, hparams

    def _get_input_targets(self, DatasetClass, metadata, dataset_hparams):
        data_loader = DatasetClass(self._hparams.batch_size, metadata, dataset_hparams)

        tensor_names = ['actions', 'images', 'states']
        if 'annotations' in data_loader:
            tensor_names = ['actions', 'images', 'states', 'annotations']

        self._tensor_multiplexer = MultiplexedTensors(data_loader, tensor_names)
        loaded_tensors = [self._tensor_multiplexer[k] for k in tensor_names]
        
        self._real_annotations = None
        assert loaded_tensors[1].get_shape().as_list()[2] == 1, "loader assumes one (potentially random) camera will be loaded in each example!"
        self._real_images = loaded_tensors[1] = loaded_tensors[1][:, :, 0]              # grab cam 0 for images
        if 'annotations' in data_loader:
            self._real_annotations = loaded_tensors[3] = loaded_tensors[3][:, :, 0]     # grab cam 0 for annotations
        
        inputs, targets = {'actions': loaded_tensors[0]}, {}
        for k, v in zip(tensor_names[1:], loaded_tensors[1:]):
            inputs[k], targets[k] = v[:, :-1], v

        self._data_loader = data_loader
        return inputs, targets
    
    def _make_dataloaders(self, config):
        DatasetClass = self._get_dataset_class(self.dataset_hparams.pop('dataset'))
        sources, self.dataset_hparams['source_selection_probabilities'] = self._init_sources()
        
        inputs, targets = self._get_input_targets(DatasetClass, sources, self.dataset_hparams)
        return inputs, targets
    
    def _default_source_hparams(self):
        return {
            'data_directory': './',
            'source_prob': None,
            'balance_by_attribute': ['robot']             # split data source into multiple sources where for each source meta[attr] == a, (e.g all examples in one source come from a specific robot)
        }

    def _init_sources(self):
        loaded_metadata = {}
        sources, source_probs = [], []

        for source in self._batch_config:
            source_hparams = self._default_source_hparams()
            source_hparams.update(source)
            dir_path = os.path.realpath(os.path.expanduser(source_hparams['data_directory']))
            meta_data = loaded_metadata[dir_path] = loaded_metadata.get(dir_path, load_metadata(dir_path))
            
            for k, v in source_hparams.items():
                if k not in self._default_source_hparams():
                    if k == 'object_classes':
                        meta_data = meta_data.select_objects(v)
                    elif isinstance(v, (list, tuple)):
                        meta_data = meta_data[meta_data[k].frame.isin(v)]
                    else:
                        meta_data = meta_data[meta_data[k] == v]
                    assert len(meta_data), "filters created empty data source!"
            
            if source_hparams['balance_by_attribute']:
                meta_data = [meta_data]
                for k in source_hparams['balance_by_attribute']:
                    new_data = []
                    for m in meta_data:
                        unique_elems = m[k].frame.unique().tolist()
                        new_data.extend([m[m[k] == u] for u in unique_elems])
                    meta_data = new_data
                
                if source_hparams['source_prob']:
                    new_prob = source_hparams['source_prob'] / float(len(meta_data))
                    source_hparams['source_prob'] = [new_prob for _ in range(len(meta_data))]
                else:
                    source_hparams['source_prob'] = [None for _ in range(len(meta_data))]
                
                sources.extend(meta_data)
                source_probs.extend(source_hparams['source_prob'])
            else:
                source_probs.append(source_hparams['source_prob'])
                sources.append(meta_data)

        if any([s is not None for s in source_probs]):
            set_probs = [s for s in source_probs if s is not None]
            assert all([0 <= s <= 1 for s in set_probs]) and sum(set_probs) <= 1, "invalid probability distribution!"
            if len(set_probs) != len(source_probs):
                remainder_prob = (1.0 - sum(set_probs)) / (len(source_probs) - len(set_probs))
                for i in range(len(source_probs)):
                    if source_probs[i] is None:
                        source_probs[i] = remainder_prob
        else:
            source_probs = None

        return sources, source_probs

    def get_batch(self, sess, mode='val'):
        return sess.run([self._inputs, self._targets], feed_dict=self._tensor_multiplexer.get_feed_dict(mode))


def _resize(image_batch, image_size):
    B = image_batch.shape[0]
    old_height, old_width = image_batch.shape[-3:-1]
    image_batch = image_batch.reshape((-1, old_height, old_width, 3))
    image_batch = (image_batch * 255).astype(np.uint8)

    target_height, target_width = image_size
    out_images = np.zeros((image_batch.shape[0], target_height, target_width, 3), dtype=np.uint8)

    resize_method = cv2.INTER_CUBIC
    if target_height * target_width < old_height * old_width:
        resize_method = cv2.INTER_AREA

    for t, img in enumerate(image_batch):
        if (old_height, old_width) == (target_height, target_width):
            out_images[t] = img
        else:
            out_images[t] = cv2.resize(img, (target_width, target_height), interpolation=resize_method)

    return out_images.reshape((B, -1, target_height, target_width, 3)) / 255.0


def get_prediction_batches(dataset, sess, prediction_model, inverse_model, mode='val'):
    batch = dataset.get_batch(sess, mode)
    actions = batch[0]['actions']
    states, images = [batch[1][x] for x in ('states', 'images')]
    context = {
            "context_frames": _resize(images[:, :prediction_model.n_context], prediction_model.img_size)[:, :, None],
            "context_actions": actions[:, :prediction_model.n_context - 1],
            "context_states": states[:, :prediction_model.n_context]
    }
    real_actions = actions[:, prediction_model.n_context - 1:]
    real_prediction_batch = {'context_tensors': context, 'action_tensors': {'actions':real_actions}}

    start, goal = images[:, prediction_model.n_context - 1], images[:, -1]
    inv_actions = inverse_model(start, goal)
    if inv_actions.shape[1] < real_actions.shape[1]:
        inv_actions = np.concatenate((inv_actions, np.zeros((inv_actions.shape[0], real_actions.shape[1] - inv_actions.shape[1], inv_actions.shape[-1]))), axis=1)
    else:
        inv_actions = inv_actions[:, :real_actions.shape[1]]

    import pdb; pdb.set_trace()

    return real_prediction_batch, {'context_tensors': context, 'action_tensors': {'actions': inv_actions}}
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_file', type=str, help='path to YAML experiment config file')
    parser.add_argument('inverse_checkpoint', type=str, help="path to inverse model checkpoint folder")
    parser.add_argument('prediction_checkpoint', type=str, help="path to video prediction model checkpoint folder")
    parser.add_argument('--N', type=int, help="number of batches to run", default=1)
    args = parser.parse_args()
    args.experiment_file = os.path.expanduser(args.experiment_file)
    args.inverse_checkpoint = os.path.expanduser(args.inverse_checkpoint)
    args.prediction_checkpoint = os.path.expanduser(args.prediction_checkpoint)

    config = parse_config(args.experiment_file)
    config.pop('train_class', None)
    
    inverse_model = ActionInferenceInterface(args.inverse_checkpoint, {"run_batch_size": 16})
    prediction_model = VPredEvaluation(args.prediction_checkpoint, {"run_batch_size": 16, 'tile_context': False})
    config['loader_hparams']['load_T'] = max(config['loader_hparams'].get('load_T', 0) + prediction_model.n_context, prediction_model.sequence_length)
    dataset = DataLoader(config)
    
    s = tf.Session()
    s.run(tf.global_variables_initializer())

    [model.set_session(s) for model in (prediction_model, inverse_model)]
    [model.restore() for model in (prediction_model, inverse_model)]

    for n in range(args.N):
        real_batch, inv_batch = get_prediction_batches(dataset, s, prediction_model, inverse_model)
        real_act_frames = prediction_model(**real_batch)['predicted_frames']
        pred_act_frames = prediction_model(**inv_batch)['predicted_frames']

        import pdb; pdb.set_trace()
        print(real_act_frames)
