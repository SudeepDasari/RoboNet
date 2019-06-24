from ray.tune.logger import TFLogger
from robonet.video_prediction.utils.encode_img import construct_image_tile
from robonet.video_prediction.utils.ffmpeg_gif import encode_gif
import numpy as np
import tensorflow as tf


class GIFLogger(TFLogger):
    def on_result(self, result):
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
