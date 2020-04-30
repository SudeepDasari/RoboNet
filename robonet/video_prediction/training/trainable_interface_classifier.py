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
from .trainable_interface import VPredTrainable

class VPredTrainableClassifier(VPredTrainable):

    def _get_input_targets(self, DatasetClass, metadata, dataset_hparams):
        data_loader = DatasetClass(self._hparams.batch_size, metadata, dataset_hparams)

        tensor_names = ['actions', 'images', 'states', 'finger_sensor']

        if 'annotations' in data_loader:
            tensor_names.append('annotations')

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
