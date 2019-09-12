from robonet.video_prediction.training.trainable_interface import VPredTrainable
from robonet.inverse_model.models import get_models
import time
from tensorflow.contrib.training import HParams
from robonet.datasets.util.tensor_multiplexer import MultiplexedTensors


class InverseTrainable(VPredTrainable):
    def _get_model_class(self, model_name):
        return get_models(model_name)

    def _default_hparams(self):
        default_dict = {
            'batch_size': 16,
            'restore_dir': '',
            'n_gpus': 1,
            'scalar_summary_freq': 100,
            'train_fraction': 0.9,
            'val_fraction': 0.05,
            'max_to_keep': 3,
            'max_steps': 300000,
        }
        return HParams(**default_dict)
    
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
        
        inputs, targets = {}, {'actions': loaded_tensors[0]}
        for k, v in zip(tensor_names[1:], loaded_tensors[1:]):
            inputs[k] = v

        self._data_loader = data_loader
        return inputs, targets

    def _train(self):
            itr = self.iteration
            
            # no need to increment itr since global step is incremented by train_op
            loss, train_op = self._estimator.loss, self._estimator.train_op
            fetches = {'global_step': itr}

            start = time.time()
            train_loss = self.sess.run([loss, train_op], feed_dict=self._tensor_multiplexer.get_feed_dict('train'))[0]
            fetches['metric/step_time'] = time.time() - start
            fetches['metric/loss/train'] = train_loss

            if itr % self._hparams.scalar_summary_freq == 0:
                fetches['metric/loss/val'] = self.sess.run(loss, feed_dict=self._tensor_multiplexer.get_feed_dict('val'))
                for name in ['train', 'val']:
                    metrics = self.sess.run(self._scalar_metrics, feed_dict=self._tensor_multiplexer.get_feed_dict(name))
                    for key, value in metrics.items():
                        fetches['metric/{}/{}'.format(key, name)] = value
    
            fetches['done'] = itr >= self._hparams.max_steps
            
            if self._hparams.restore_dir and not self._restore_logs:
                fetches['restore_logs'] = self._hparams.restore_dir
                self._restore_logs = True
    
            return fetches
