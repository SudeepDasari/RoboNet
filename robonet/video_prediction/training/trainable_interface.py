from ray.tune import Trainable
import tensorflow as tf
from robonet.datasets import get_dataset_class, MultiplexedTensors, load_metadata
from robonet.video_prediction.models import get_model_fn
from robonet.video_prediction.utils import tf_utils
import numpy as np
import os
from tensorflow.contrib.training import HParams


class VPredTrainable(Trainable):
    def _setup(self, config):
        # run hparams are passed in through config dict
        dataset_hparams, model_hparams, self._hparams = self._extract_hparams(config)
        DatasetClass, model_fn = get_dataset_class(dataset_hparams.pop('dataset')), get_model_fn(model_hparams.pop('model'))

        metadata = self._filter_metadata(load_metadata(config['data_directory']))
        dataset = DatasetClass(config.pop('batch_size'), metadata=metadata, hparams=dataset_hparams)

        inputs, targets = self._get_input_targets(dataset)
        self._estimator = model_fn(self._hparams.n_gpus, self._hparams.graph_type, inputs, targets, tf.estimator.ModeKeys.TRAIN, model_hparams)
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        self._global_step = tf.train.get_or_create_global_step()
        self.saver = tf.train.Saver(max_to_keep=self._hparams.max_to_keep)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print("parameter_count =", self.sess.run(parameter_count))

    def _extract_hparams(self, config):
        """
        Grabs and (optionally) modifies hparams
        """
        dataset_hparams, model_hparams = config.pop('dataset_hparams', {}), config.pop('model_hparams', {})
        hparams = self._default_hparams().override_from_dict(config)
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
            'data_directory': './',
            'n_gpus': 1,
            'pad_amount': 2,
            'val_summary_freq': 100,
            'image_summary_freq': 1000,
            'train_fraction': 0.9,
            'val_fraction': 0.05,
            'max_to_keep': 3,
            'robot': '',
            'action_primitive': '',
            'filter_adim': 0,
            'max_steps': 300000
        }
        return HParams(**default_dict)

    def _filter_metadata(self, metadata):
        """
        filters metadata based on configuration file
            - overwrite/modify for more expressive data loading
        """
        if self._hparams.action_primitive:
            metadata = metadata[metadata['primitives'] == self._hparams.action_primitive]
        if self._hparams.robot:
            metadata = metadata[metadata['robot'] == self._hparams.robot]
        if self._hparams.filter_adim:
            metadata = metadata[metadata['adim'] == self._hparams.filter_adim]
        assert len(metadata), "no data matches filters!"
        return metadata

    def _get_input_targets(self, data_loader):
        assert data_loader.hparams.get('load_random_cam', False), "function assumes loader will grab one random camera feed in multi-cam object"
        self._tensor_multiplexer = MultiplexedTensors(data_loader, ['images', 'actions', 'states'])
        images, actions, states = [self._tensor_multiplexer[k] for k in ['images', 'actions', 'states']]
        images = images[:, :, 0]          # grab cam 0
        
        inputs, targets = {'actions': actions}, {}
        for k, v in zip(['states', 'images'], [states, images]):
            inputs[k], targets[k] = v[:, :-1], v
        self._real_images = images

        return inputs, targets

    def _train(self):
        itr = self.iteration
        
        # no need to increment itr since global step is incremented by train_op
        loss, train_op, predicted = self._estimator.loss, self._estimator.train_op, self._estimator.predictions
        
        fetches = {'global_step': itr}
        fetches['metric/train/loss'] = self.sess.run([loss, train_op], feed_dict=self._tensor_multiplexer.train)[0]
        
        if itr % self._hparams.image_summary_freq == 0:
            for name, fetch in zip(['train', 'val'], [self._tensor_multiplexer.train, self._tensor_multiplexer.val]):
                real, pred = [(x * 255).astype(np.uint8) for x in self.sess.run([self._real_images, predicted], feed_dict=fetch)]
                pred = np.concatenate([pred[:, 0][:, None] for _ in range(real.shape[1] - pred.shape[1])] + [pred], axis=1)

                image_summary_tensors = []
                for tensor in [real, pred]:
                    height_pad = np.zeros((tensor.shape[0], tensor.shape[1], self._hparams.pad_amount, tensor.shape[-2], tensor.shape[-1]), dtype=np.uint8)
                    tensor = np.concatenate((height_pad, tensor, height_pad), axis=-3)
                    width_pad = np.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2], self._hparams.pad_amount, tensor.shape[-1]), dtype=np.uint8)
                    tensor = np.concatenate((width_pad, tensor, width_pad), axis=-2)
                    image_summary_tensors.append(tensor)
        
                fetches['metric/{}/image_summary'.format(name)] = np.concatenate(image_summary_tensors, axis=2)

        if itr % self._hparams.val_summary_freq == 0:
            fetches['metric/val/loss'] = self.sess.run(loss, feed_dict=self._tensor_multiplexer.val)

        fetches['done'] =  itr >= self._hparams.max_steps
        
        return fetches

    def _save(self, checkpoint_dir):
        return self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model'), global_step=self.iteration) + '.meta'
        

    def _restore(self, checkpoints):
        # possibly restore from multiple checkpoints. useful if subset of weights
        # (e.g. generator or discriminator) are on different checkpoints.
        if not isinstance(checkpoints, (list, tuple)):
            checkpoints = [checkpoints]
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
