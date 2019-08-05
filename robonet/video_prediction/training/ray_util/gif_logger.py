import ray.cloudpickle as cloudpickle
from ray.tune.logger import TFLogger, Logger
from robonet.video_prediction.utils.encode_img import construct_image_tile
from robonet.video_prediction.utils.ffmpeg_gif import encode_gif
import numpy as np
import tensorflow as tf
import os
import glob
import shutil
import sys
import pickle as pkl


class TFImageLogger(TFLogger):
    def on_result(self, result):
        restore_dir = result.pop('restore_logs', False)
        if restore_dir:
            # close the old file_writer
            self._file_writer.close()
            
            # copy log events to new directory
            event_dir = restore_dir.split('/checkpoint')[0]
            event_file = glob.glob('{}/events.out.*'.format(event_dir))[0]
            new_path = '{}/{}'.format(self.logdir,event_file.split('/')[-1])
            assert os.path.isfile(event_file), "even logs don't exist!"
            shutil.copyfile(event_file, new_path)

            # initialize a new file writer
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


class GIFLogger(Logger):
    def _init(self):
        self._save_dir = os.path.join(self.logdir, 'metrics')
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
    
        self._metric_file = os.path.join(self._save_dir, 'metric_summaries.pkl')
        if os.path.exists(self._metric_file):
            self._metric_logs = pkl.load(open(self._metric_file, 'rb'))
        else:
            self._metric_logs = {}
        self._image_logs = {}
    
    def flush(self):
        with open(self._metric_file, 'wb') as f:
            cloudpickle.dump(self._metric_logs, f)
        
        if self._image_logs:
            img_dir = os.path.join(self._save_dir, 'images')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            for metric_name, summaries in self._image_logs.items():
                for step, encoding_type, encoded_im in summaries:
                    assert encoding_type == 'GIF'
                    file_name = '{}/{}_summary_{}.gif'.format(img_dir, metric_name, step)
                    with open(os.path.join(self._save_dir, file_name), 'wb') as f:
                        f.write(encoded_im)
            self._image_logs = {}
    
    def on_result(self, result):
        global_step = result['global_step']
        report_step = False
        for k, v in result.items():
            if 'metric/' not in k or 'step_time' in k:
                continue
                        
            report_step = True
            tag = '_'.join(k.split('/')[1:])
            if isinstance(v, np.ndarray):
                assert v.dtype == np.uint8 and len(v.shape) >= 4, 'assume np arrays are  batched image data'
                self._image_logs[tag] = self._image_logs.get(tag, []) + [(global_step, 'GIF', encode_gif(construct_image_tile(v), 4))]
            else:
                self._metric_logs[tag] = self._metric_logs.get(tag, []) + [v]
            
        if report_step:
            self._metric_logs['global_step'] = self._metric_logs.get('global_step', []) + [global_step]

        self.flush()
