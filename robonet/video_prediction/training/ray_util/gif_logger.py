from ray.tune.logger import TFLogger
from robonet.video_prediction.utils.encode_img import construct_image_tile
from robonet.video_prediction.utils.ffmpeg_gif import encode_gif
import numpy as np
import tensorflow as tf
import os
import glob
import shutil


class GIFLogger(TFLogger):
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
