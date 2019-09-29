from ray.tune import Trainable
import pdb
import tensorflow as tf
from robonet.datasets import get_dataset_class, load_metadata
from robonet.video_prediction.models import get_model
from robonet.video_prediction.utils import tf_utils
import numpy as np
import os
from tensorflow.contrib.training import HParams
from .util import pad_and_concat, render_dist, pad, stbmajor
import time
import glob
from robonet.datasets.util.tensor_multiplexer import MultiplexedTensors
import yaml
import shutil
from robonet.video_prediction.utils.encode_img import construct_image_tile
from robonet.video_prediction.utils.ffmpeg_gif import encode_gif
import copy


class VPredTrainable(Trainable):
    def _setup(self, config):
        self._base_config = copy.deepcopy(config)
        # run hparams are passed in through config dict
        self.dataset_hparams, self.model_hparams, self._hparams = self._extract_hparams(config)
        inputs, targets = self._make_dataloaders(config)

        self._model_name = self.model_hparams.pop('model')
        PredictionModel = self._get_model_class(self._model_name)
        self._model = model = PredictionModel(self._data_loader.hparams, self._hparams.n_gpus, self._hparams.graph_type, False)
        est, s_m, t_m = model.model_fn(inputs, targets, tf.estimator.ModeKeys.TRAIN, self.model_hparams)
        self._estimator, self._scalar_metrics, self._tensor_metrics = est, s_m, t_m
        try:
            parameter_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            print("parameter_count =", parameter_count)
        except TypeError:
            pass

        self._global_step = tf.train.get_or_create_global_step()
        self.saver = tf.train.Saver(max_to_keep=self._hparams.max_to_keep)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.annotation_modes = None

        if self._hparams.restore_dir:
            meta_file = glob.glob(self._hparams.restore_dir + '/*.meta')
            self._restore(meta_file[0])
            self._restore_logs = False
        self._file_writer = None

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
    
    def _get_model_class(self, class_name):
        return get_model(class_name)

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
        assert loaded_tensors[1].get_shape().as_list()[2] == 1, "loader assumes one (potentially random) camera will be loaded in each example!"

        self._real_annotations = None
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

    def _train(self):
        itr = self.iteration
        
        # no need to increment itr since global step is incremented by train_op
        loss, train_op = self._estimator.loss, self._estimator.train_op
        fetches = {'global_step': itr}

        start = time.time()
        train_loss = self.sess.run([loss, train_op], feed_dict=self._tensor_multiplexer.get_feed_dict('train'))[0]
        fetches['metric/step_time'] = time.time() - start
       # import pdb; pdb.set_trace()
        if itr % self._hparams.image_summary_freq == 0 or itr % self._hparams.scalar_summary_freq == 0 or self.annotation_modes is None:
            img_summary_get_ops = {'real_images':self._real_images,
                                   'pred_frames':self._tensor_metrics['pred_frames'],
                                   }
            if self._real_annotations is not None:
                img_summary_get_ops.update({'real_annotation':self._real_annotations,
                                            'pred_distrib':self._tensor_metrics['pred_distrib']})
                self.annotation_modes = [m for m in self._tensor_multiplexer.modes if '_annotated' in m]
            else:
                self.annotation_modes = []

        if itr % self._hparams.image_summary_freq == 0:
            if 'pred_targets' in self._tensor_metrics:  # used for embedding model
                img_summary_get_ops['pred_targets'] = self._tensor_metrics['pred_targets']
                img_summary_get_ops['pred_target_dists'] = self._tensor_metrics['pred_target_dists']
                img_summary_get_ops['inference_images'] = self._tensor_metrics['inference_images']
            
            for name in ['train', 'val'] + self.annotation_modes:
                fetch_mode = self._tensor_multiplexer.get_feed_dict(name)
                fetched_npy = self.sess.run(img_summary_get_ops, feed_dict=fetch_mode)

                if self._real_annotations is not None and '_annotated' in name:
                    if 'pred_targets' in self._tensor_metrics:  # used for embedding model
                        dists = (stbmajor(fetched_npy['pred_target_dists']), fetched_npy['pred_distrib'])
                    else:
                        dists = (fetched_npy['real_annotation'], fetched_npy['pred_distrib'])
                    for o in range(len(dists)):
                        dist_name = 'robot'
                        if o > 0:
                            dist_name = 'object{}'.format(o)
                        real_dist, pred_dist = [render_dist(x[:, :, :, :, o]) for x in dists]
                        fetches['metric/{}_pixel_warp/{}'.format(dist_name, name)] = pad_and_concat(real_dist, pred_dist, self._hparams.pad_amount)
                else:
                    real_img, pred_img = fetched_npy['real_images'], fetched_npy['pred_frames']
                    if real_img.shape[0] == pred_img.shape[0]*2:  # if using different trajectories for inference and prediction
                        fetches['metric/image_summary/{}_all'.format(name)] = pad(fetched_npy['real_images'], self._hparams.pad_amount)
                        # real_img_inf, real_img = split_model_inference(real_img, params=self.model_hparams)
                        fetches['metric/image_summary/{}_inference'.format(name)] = pad(stbmajor(fetched_npy['inference_images']), self._hparams.pad_amount)
                        fetches['metric/image_summary/{}'.format(name)] = pad_and_concat(stbmajor(fetched_npy['pred_targets']), fetched_npy['pred_frames'], self._hparams.pad_amount)
                    else:  # used for everything else:
                        fetches['metric/image_summary/{}'.format(name)] = pad_and_concat(fetched_npy['real_images'], fetched_npy['pred_frames'], self._hparams.pad_amount)

        if itr % self._hparams.scalar_summary_freq == 0:
            fetches['metric/loss/train'] = train_loss
            fetches['metric/loss/val'] = self.sess.run(loss, feed_dict=self._tensor_multiplexer.get_feed_dict('val'))
            for name in ['train', 'val'] + self.annotation_modes:
                metrics = self.sess.run(self._scalar_metrics, feed_dict=self._tensor_multiplexer.get_feed_dict(name))
                for key, value in metrics.items():
                    if 'pixel' in key and '_annotated' not in name:
                        # doesn't log pixel metrics for trajs which don't have pixels
                        continue
                    elif 'pixel' not in key and '_annotated' in name:
                        # don't log other metrics (like l1_loss) for debug modes
                        continue

                    fetches['metric/{}/{}'.format(key, name)] = value

        fetches['done'] = itr >= self._hparams.max_steps
        
        self._tf_log(fetches)

        return fetches

    def _save(self, checkpoint_dir):
        dataset_params = self._model.data_hparams.values()
        model_params = self._model.model_hparams.values()
        model_params['model'] = self._model_name
        model_params['graph_type'] = self._hparams.graph_type
        model_params['scope_name'] = self._model.scope_name

        with open(os.path.join(checkpoint_dir, 'params.yaml'), 'w') as f:
            yaml.dump({'model': model_params, 'dataset': dataset_params}, f)
        with open(os.path.join(checkpoint_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self._base_config, f)
        
        return self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model'), global_step=self.iteration) + '.meta'

    def _restore(self, checkpoints):
        # possibly restore from multiple checkpoints. useful if subset of weights
        # (e.g. generator or discriminator) are on different checkpoints.
        checkpoints = [checkpoints.split('.meta')[0]]
        # automatically skip global_step if more than one checkpoint is provided
        skip_global_step = len(checkpoints) > 1
        savers = []
        for checkpoint in checkpoints:
            print("creating restore saver from checkpoint %s" % checkpoint)
            saver, _ = tf_utils.get_checkpoint_restore_saver(checkpoint, skip_global_step=skip_global_step)
            savers.append(saver)
        restore_op = [saver.saver_def.restore_op_name for saver in savers]
        return self.sess.run(restore_op)

    @property
    def iteration(self):
        return self.sess.run(self._global_step)

    def _tf_log(self, result):
        if self._hparams.restore_dir and not self._restore_logs:
            # close the old file_writer
            self._file_writer.close()
            
            # copy log events to new directory
            event_dir = self._hparams.restore_dir.split('/checkpoint')[0]
            event_file = glob.glob('{}/events.out.*'.format(event_dir))[0]
            new_path = '{}/{}'.format(self.logdir,event_file.split('/')[-1])
            assert os.path.isfile(event_file), "even logs don't exist!"
            shutil.copyfile(event_file, new_path)

            # initialize a new file writer
            self._restore_logs = True

        if self._file_writer is None:
            self._file_writer = tf.summary.FileWriter(self.logdir)
        
        global_step = result['global_step']

        for k, v in result.items():
            if 'metric/' not in k:
                continue
            
            tag = '/'.join(k.split('/')[1:])
            summary = tf.Summary()
            if isinstance(v, np.ndarray):
                assert v.dtype == np.uint8 and len(v.shape) >= 4, 'assume np arrays are  batched image data'
                image = tf.Summary.Image()
                image.height = v.shape[-3]
                image.width = v.shape[-2]
                image.colorspace = v.shape[-1]  # 1: grayscale, 2: grayscale + alpha, 3: RGB, 4: RGBA
                image.encoded_image_string = encode_gif(construct_image_tile(v), 4)
                summary.value.add(tag=tag, image=image)
            else:
                summary.value.add(tag=tag, simple_value=v)
            self._file_writer.add_summary(summary, global_step)
        self._file_writer.flush()
