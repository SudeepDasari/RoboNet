import itertools

import tensorflow as tf

from video_prediction.models import SAVPVideoPredictionModel
from video_prediction.utils import tf_utils
from . import vgg_network
import pdb


class IndepMultiSAVPVideoPredictionModel(SAVPVideoPredictionModel):
    def get_default_hparams_dict(self):
        default_hparams = super(IndepMultiSAVPVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            num_views=1,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def tower_fn(self, inputs, targets=None):
        def merge(tuple0, tuple1, suffix):
            """merges tuple1 into tuple0"""
            assert len(tuple0) == len(tuple1)
            list0, list1 = list(tuple0), list(tuple1)
            for d0, d1 in zip(list0, list1):
                if isinstance(d0, dict):
                    for k, v in d1.items():
                        d0[k + suffix] = v
            return tuple(list0)

        for i in range(self.hparams.num_views):
            suffix = '%d' % i if i > 0 else ''
            # pdb.set_trace()
            inputs_view = {k: v for k, v in inputs.items() if not k.startswith('images') and not k.startswith('pix_distribs')}
            inputs_view['images'] = inputs['images' + suffix]
            inputs_view['pix_distribs'] = inputs['pix_distribs' + suffix]
            targets_view = targets if i == 0 else None  # assume targets correspond to the first view
            with tf.variable_scope('view%d' % i):
                outputs_view, losses_view, metrics_view = \
                    SAVPVideoPredictionModel.tower_fn(self, inputs_view, targets=targets_view)
            if i == 0:
                outputs = outputs_view
                losses = losses_view
                metrics = metrics_view
            else:
                # the non-dict elements are ignored (e.g. the tensors gen_images and gen_images_enc)
                # effectively, the returned gen_images corresponds to the first view
                outputs = merge(outputs, outputs_view, suffix)
                losses = merge(losses, losses_view, suffix)
                metrics = merge(metrics, metrics_view, suffix)

        return outputs, losses, metrics

    def restore(self, sess, checkpoints):
        vgg_network.vgg_assign_from_values_fn()(sess)

        if checkpoints:
            if not isinstance(checkpoints, (list, tuple)):
                checkpoints = [checkpoints]
            if len(checkpoints) != self.hparams.num_views:
                raise ValueError('number of checkpoints should be equal to the number of views')
            savers = []
            for i, checkpoint in enumerate(checkpoints):
                print("creating restore saver from checkpoint %s" % checkpoint)
                restore_scope = 'view%d' % i

                def restore_to_checkpoint_mapping(name):
                    name = name.split(':')[0]
                    assert name.split('/')[0] == restore_scope
                    name = '/'.join(name.split('/')[1:])
                    return name

                saver, _ = tf_utils.get_checkpoint_restore_saver(checkpoint, restore_to_checkpoint_mapping=restore_to_checkpoint_mapping, restore_scope=restore_scope)
                savers.append(saver)
            restore_op = [saver.saver_def.restore_op_name for saver in savers]
            sess.run(restore_op)
