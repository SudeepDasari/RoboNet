from ray.tune import Trainable
import tensorflow as tf
from robonet.datasets import get_dataset_class, MultiplexedTensors, load_metadata
from robonet.video_prediction.models import get_model_fn
from robonet.video_prediction.utils import tf_utils
import numpy as np
import os


class VPredTrainable(Trainable):
    def _setup(self, config):
        # run hparams are passed in through config dict
        dataset_hparams, model_hparams, graph_type, n_gpus = self._extract_hparams(config)
        self._config, self._dataset_hparams, self._model_hparams = config, dataset_hparams, model_hparams
        DatasetClass, model_fn = get_dataset_class(dataset_hparams.pop('dataset')), get_model_fn(model_hparams.pop('model'))

        metadata = self._filter_metadata(load_metadata(config['data_directory']))
        if model_hparams['num_domains'] == 1:
            dataset = DatasetClass(config.pop('batch_size'), metadata=metadata, hparams=dataset_hparams)
            print('loaded dataset!')
            inputs, targets = self._get_input_targets(dataset)
        else:
            self._tensor_multiplexers = []
            self._input_images = []
            batch_size = config.pop('batch_size')
            input_images, input_actions, input_states, target_images, target_states = [], [], [], [], []

            for i in range(model_hparams['num_domains']):
                mod_metadata = metadata[metadata['camera_configuration'] == 'sudri{}'.format(i)]
                dataset = DatasetClass(batch_size, metadata=mod_metadata, hparams=dataset_hparams)
                print('loaded dataset!')

                inputs, targets = self._get_input_targets(dataset)
                input_images.append(inputs['images'])
                input_actions.append(inputs['actions'])
                input_states.append(inputs['states'])
                target_images.append(targets['images'])
                target_states.append(targets['states'])

            input_images = tf.concat(input_images, axis=0)
            input_actions = tf.concat(input_actions, axis=0)
            input_states = tf.concat(input_states, axis=0)
            target_images = tf.concat(target_images, axis=0)
            target_states = tf.concat(target_states, axis=0)
            inputs = {'images': input_images, 'actions': input_actions, 'states': input_states}
            targets = {'images': target_images, 'states': target_states}

            self._train_feed_dict, self._val_feed_dict = {}, {}
            for t in self._tensor_multiplexers:
                self._train_feed_dict.update(t.train)
                self._val_feed_dict.update(t.val)
        self._estimator = model_fn(n_gpus, graph_type, inputs, targets, tf.estimator.ModeKeys.TRAIN, model_hparams)

        self._global_step = tf.train.get_or_create_global_step()
        self.saver = tf.train.Saver(max_to_keep=config.get('max_to_keep', 3))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _extract_hparams(self, config):
        """
        Grabs and (optionally) modifies hparams
        """
        dataset_hparams, model_hparams = config.pop('dataset_hparams', {}), config.pop('model_hparams', {})
        graph_type = model_hparams.pop('graph')

        if 'train_frac' in config:
            train_frac = config['train_frac']
            dataset_hparams['splits'] = [train_frac, 0.95 - train_frac, 0.05]

        self._val_summary_freq = config.get('val_summary_freq', 100)
        self._image_summary_freq = config.get('image_summary_freq', 5000)

        if 'sequence_length' in model_hparams and 'load_T' not in dataset_hparams:
            dataset_hparams['load_T'] = model_hparams['sequence_length'] + 1
        return dataset_hparams, model_hparams, graph_type, config.get('n_gpus', 1)

    def _filter_metadata(self, metadata):
        """
        filters metadata based on configuration file
            - overwrite/modify for more expressive data loading
        """
        return metadata[metadata['adim']  == 4]
    
    def _get_input_targets(self, data_loader):
        assert data_loader.hparams.get('load_random_cam', False), "function assumes loader will grab one random camera feed in multi-cam object"
        # self._tensor_multiplexer = MultiplexedTensors(data_loader, ['images', 'actions', 'states'])
        # images, actions, states = [self._tensor_multiplexer[k] for k in ['images', 'actions', 'states']]
        t = MultiplexedTensors(data_loader, ['images', 'actions', 'states'])
        images, actions, states = [t[k] for k in ['images', 'actions', 'states']]
        self._tensor_multiplexers.append(t)
        images = images[:, :, 0]          # grab cam 0
        
        inputs, targets = {'actions': actions}, {}
        for k, v in zip(['states', 'images'], [states, images]):
            inputs[k], targets[k] = v[:, :-1], v
        # self._input_images = inputs['images']
        self._input_images.append(inputs['images'])
        return inputs, targets

    def _train(self):
        itr = self.iteration
        
        # no need to increment itr since global step is incremented by train_op
        loss, train_op, predicted = self._estimator.loss, self._estimator.train_op, self._estimator.predictions
        input_images = tf.concat(self._input_images, axis=0)
        
        fetches = {'global_step': itr}
        # fetches['metric/train/loss'] = self.sess.run([loss, train_op], feed_dict=self._tensor_multiplexer.train)[0]
        fetches['metric/train/loss'] = self.sess.run([loss, train_op], feed_dict=self._train_feed_dict)[0]
        
        if itr % self._image_summary_freq == 0:
            # for name, fetch in zip(['train', 'val'], [self._tensor_multiplexer.train, self._tensor_multiplexer.val]):
            for name, fetch in zip(['train', 'val'], [self._train_feed_dict, self._val_feed_dict]):
                real, pred = [(x * 255).astype(np.uint8) for x in self.sess.run([input_images, predicted], feed_dict=fetch)]
                fetches['metric/{}/input_images'.format(name)] = real
                fetches['metric/{}/predicted_images'.format(name)] = pred

        if itr % self._val_summary_freq == 0:
            # fetches['metric/val/loss'] = self.sess.run(loss, feed_dict=self._tensor_multiplexer.val)
            fetches['metric/val/loss'] = self.sess.run(loss, feed_dict=self._val_feed_dict)

        return fetches

    def _save(self, checkpoint_dir):
        return self.saver.save(
            self.sess, os.path.join(checkpoint_dir, 'model'),
            global_step=self.iteration)

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
