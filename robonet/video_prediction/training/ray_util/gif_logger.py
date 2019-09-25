import ray.cloudpickle as cloudpickle
from ray.tune.logger import Logger
import numpy as np
import os
import pickle as pkl
from robonet.video_prediction.utils.ffmpeg_gif import encode_gif
from robonet.video_prediction.utils.encode_img import construct_image_tile


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
