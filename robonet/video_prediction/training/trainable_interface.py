from ray.tune import Trainable
import pdb
import tensorflow as tf
from robonet.datasets import get_dataset_class, MultiplexedTensors, load_metadata
from robonet.video_prediction.models import get_model_fn
from robonet.video_prediction.utils import tf_utils
import numpy as np
import os
from tensorflow.contrib.training import HParams
from .util import pad_and_concat, render_dist, pad, stbmajor
import time
import glob


class VPredTrainable(Trainable):
    def _setup(self, config):
        # run hparams are passed in through config dict
        self.dataset_hparams, self.model_hparams, self._hparams = self._extract_hparams(config)
        DatasetClass, model_fn = get_dataset_class(self.dataset_hparams.pop('dataset')), get_model_fn(self.model_hparams.pop('model'))

        metadata = self._filter_metadata(load_metadata(config['data_directory']))

        self._real_images = []
        inputs, targets = self._get_input_targets(DatasetClass, metadata, self.dataset_hparams)

        self._real_images = tf.concat(self._real_images, axis=0)
        self._estimator, self._scalar_metrics, self._tensor_metrics = model_fn(self._hparams.n_gpus, self._hparams.graph_type, 
                                                    False, inputs, targets, tf.estimator.ModeKeys.TRAIN, self.model_hparams)
        self._parameter_count = parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        self._global_step = tf.train.get_or_create_global_step()
        self.saver = tf.train.Saver(max_to_keep=self._hparams.max_to_keep)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        if self._hparams.restore_dir:
            meta_file = glob.glob(self._hparams.restore_dir + '/*.meta')
            self._restore(meta_file[0])
            self._restore_logs = False
    
        print("parameter_count =", self.sess.run(parameter_count))

    def _extract_hparams(self, config):
        """
        Grabs and (optionally) modifies hparams
        """
        dataset_hparams, model_hparams = config.pop('dataset_hparams', {}), config.pop('model_hparams', {})
        if 'sub_batch_size' in dataset_hparams:
            model_hparams['sub_batch_size'] = dataset_hparams['sub_batch_size']
        hparams = self._default_hparams().override_from_dict(config)
        model_hparams['batch_size'] = hparams.batch_size
        hparams.graph_type = model_hparams.pop('graph_type')                      # required key which tells model which graph to load
        assert hparams.max_steps > 0, "max steps must be positive!"

        if 'splits' not in dataset_hparams:
            dataset_hparams['splits'] = [hparams.train_fraction, hparams.val_fraction, 1 - hparams.val_fraction - hparams.train_fraction]
            assert all([x >= 0 for x in dataset_hparams['splits']]), "invalid train/val fractions!"

        if 'sequence_length' in model_hparams and 'load_T' not in dataset_hparams:
            dataset_hparams['load_T'] = model_hparams['sequence_length'] + 1
        
        return dataset_hparams, model_hparams, hparams
    
    def _default_hparams(self):
        default_dict = {
            'batch_size': 16,
            'restore_dir': '',
            'data_directory': './',
            'n_gpus': 1,
            'pad_amount': 2,
            'scalar_summary_freq': 100,
            'image_summary_freq': 1000,
            'train_fraction': 0.9,
            'val_fraction': 0.05,
            'max_to_keep': 3,
            'robot': '',
            'action_primitive': '',
            'filter_adim': 0,
            'max_steps': 300000,
            'balance_across_robots': False
        }
        return HParams(**default_dict)

    def _filter_metadata(self, metadata):
        """
        filters metadata based on configuration file
            - overwrite/modify for more expressive data loading
        """
        if self._hparams.action_primitive:
            metadata = metadata[metadata['primitives'] == self._hparams.action_primitive]
        if self._hparams.filter_adim:
            metadata = metadata[metadata['adim'] == self._hparams.filter_adim]

        if self._hparams.balance_across_robots or self._hparams.robot == 'all_balanced':
            assert not self._hparams.robot or self._hparams.robot == 'all_balanced', "can't balance across one robot!"
            unique_robots = metadata['robot'].frame.unique().tolist()
            all_metadata = metadata
            metadata = [all_metadata[all_metadata['robot'] == r] for r in unique_robots]  
        elif self._hparams.robot:
            metadata = metadata[metadata['robot'] == self._hparams.robot]
        
        if 'train_ex_per_source' in self.dataset_hparams:
            if not isinstance(self.dataset_hparams['train_ex_per_source'], list):
                print('train_ex_per_source is not a list! Automatically broadcasting...')
                if isinstance(metadata, list):
                    self.dataset_hparams['train_ex_per_source'] = [self.dataset_hparams['train_ex_per_source'] for _ in range(len(metadata))]
                else:
                    self.dataset_hparams['train_ex_per_source'] = [self.dataset_hparams['train_ex_per_source']]
        return metadata

    def _get_input_targets(self, DatasetClass, metadata, dataset_hparams):
        data_loader = DatasetClass(self._hparams.batch_size, metadata, dataset_hparams)
        assert data_loader.hparams.get('load_random_cam', False), "function assumes loader will grab one random camera feed in multi-cam object"

        tensor_names = ['actions', 'images', 'states']
        if 'annotations' in data_loader:
            tensor_names = ['actions', 'images', 'states', 'annotations']

        self._tensor_multiplexer = MultiplexedTensors(data_loader, tensor_names)
        loaded_tensors = [self._tensor_multiplexer[k] for k in tensor_names]
        
        self._real_annotations = None
        self._real_images = loaded_tensors[1] = loaded_tensors[1][:, :, 0]              # grab cam 0 for images
        if 'annotations' in data_loader:
            loaded_tensors[3] = loaded_tensors[3][:, :, 0]                              # grab cam 0 for annotations
            self._real_annotations = loaded_tensors[3]
        
        inputs, targets = {'actions': loaded_tensors[0]}, {}
        for k, v in zip(tensor_names[1:], loaded_tensors[1:]):
            inputs[k], targets[k] = v[:, :-1], v

        return inputs, targets

    def _train(self):
        itr = self.iteration
        
        # no need to increment itr since global step is incremented by train_op
        loss, train_op = self._estimator.loss, self._estimator.train_op
        
        fetches = {'global_step': itr}

        start = time.time()
        train_loss = self.sess.run([loss, train_op], feed_dict=self._tensor_multiplexer.get_feed_dict('train'))[0]
        fetches['metric/step_time'] = time.time() - start
        
        if itr % self._hparams.image_summary_freq == 0 or itr % self._hparams.scalar_summary_freq == 0:
            img_summary_get_ops = {'real_images':self._real_images,
                                   'pred_frames':self._tensor_metrics['pred_frames'],
                                   }
            if self._real_annotations is not None:
                img_summary_get_ops.update({'real_annotation':self._real_annotations,
                                            'pred_distrib':self._tensor_metrics['pred_distrib']})
                annotation_modes = [m for m in self._tensor_multiplexer.modes if '_annotated' in m]
            else:
                annotation_modes = []
        
        if itr % self._hparams.image_summary_freq == 0:
            if 'pred_targets' in self._tensor_metrics:  # used for embedding model
                img_summary_get_ops['pred_targets'] = self._tensor_metrics['pred_targets']
                img_summary_get_ops['inference_images'] = self._tensor_metrics['inference_images']
            
            for name in ['train', 'val'] + annotation_modes:
                fetch_mode = self._tensor_multiplexer.get_feed_dict(name)
                fetched_npy = self.sess.run(img_summary_get_ops, feed_dict=fetch_mode)

                if self._real_annotations is not None and '_annotated' in name:
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
            for name in ['train', 'val'] + annotation_modes:
                metrics = self.sess.run(self._scalar_metrics, feed_dict=self._tensor_multiplexer.get_feed_dict(name))
                for key, value in metrics.items():
                    if 'pixel' in key and '_annotated' not in name:
                        # doesn't log pixel metrics for trajs which don't have pixels
                        continue
                    fetches['metric/{}/{}'.format(key, name)] = value

        fetches['done'] = itr >= self._hparams.max_steps
        
        if self._hparams.restore_dir and not self._restore_logs:
            fetches['restore_logs'] = self._hparams.restore_dir
            self._restore_logs = True

        return fetches

    def _save(self, checkpoint_dir):
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
