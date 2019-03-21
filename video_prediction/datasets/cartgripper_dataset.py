import itertools
import pdb
import re
import tensorflow as tf
from .base_dataset import VideoDataset
from collections import OrderedDict
from .softmotion_dataset import SoftmotionVideoDataset


class CartgripperVideoDataset(SoftmotionVideoDataset):
    def __init__(self, *args, **kwargs):
        VideoDataset.__init__(self, *args, **kwargs)
        if self.hparams.image_view != -1:
            self.state_like_names_and_shapes['images'] = \
                '%%d/image_view%d/encoded' % self.hparams.image_view, (48, 64, 3)
        else:
            # infer name of image feature(s)
            from google.protobuf.json_format import MessageToDict
            example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
            dict_message = MessageToDict(tf.train.Example.FromString(example))
            feature = dict_message['features']['feature']
            image_names = set()
            for name in feature.keys():
                m = re.search('\d+/(\w+)/encoded', name)
                if m:
                    image_names.add(m.group(1))
            for image_name in image_names:
                m = re.search('image_view(\d+)', image_name)
                if m:
                    i = int(m.group(1))
                    suffix = '%d' % i if i > 0 else ''
                    self.state_like_names_and_shapes['images' + suffix] = \
                        '%%d/%s/encoded' % image_name, (48, 64, 3)

        if self.hparams.use_state:
            self.state_like_names_and_shapes['states'] = '%d/endeffector_pos', (self.hparams.sdim,)
        self.action_like_names_and_shapes['actions'] = '%d/action', (self.hparams.adim,)
        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        default_hparams = super(CartgripperVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=15,
            time_shift=3,
            use_state=True,
            sdim=6,
            adim=3,
            image_view=-1,
            autograsp=-1, # only take first n dimensions of action vector
            ignore_touch=False,
            saturate_touch=True,
            touch_no_state=False,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))


    def parser(self, serialized_example):
        state_like_seqs, action_like_seqs = super(CartgripperVideoDataset, self).parser(serialized_example)
        if 'actions' in action_like_seqs:
            if self.hparams.autograsp != -1:
                assert action_like_seqs['actions'].get_shape().as_list()[1] == 5
                action_like_seqs['actions'] = action_like_seqs['actions'][:,:self.hparams.autograsp]

        if 'states' in state_like_seqs:
            if self.hparams.ignore_touch:
                state_like_seqs['states'] = state_like_seqs['states'][:,:-2]

            if self.hparams.saturate_touch and not self.hparams.ignore_touch:
                assert state_like_seqs['states'].get_shape().as_list()[1] == 7
                state = state_like_seqs['states'][:,:5]
                touch = state_like_seqs['states'][:,5:]
                touch = tf.nn.sigmoid(touch)

                if self.hparams.touch_no_state:
                    state_like_seqs['states'] = touch
                else:
                    state_like_seqs['states'] = tf.concat([state, touch], axis=1)

        return state_like_seqs, action_like_seqs
