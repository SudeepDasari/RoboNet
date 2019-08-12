"""
Boiled down version of SAVP model from https://github.com/alexlee-gk/video_prediction
"""
from robonet.video_prediction.models.base_model import BaseModel
from robonet.video_prediction.utils import tf_utils
import tensorflow as tf
from collections import OrderedDict
from robonet.video_prediction import losses
from robonet.video_prediction import metrics
from robonet.video_prediction.models.deterministc_embedding_utils import onestep_encoder_fn, average_and_repeat, split_model_inference
import logging


def host_summary_fn(summary_dir, summary_queue_len, image_summary_freq, **summary_dict):
    gs = summary_dict.pop('global_step')[0]               # the 0 index here is crucial, will error on TPU otherwise
    real_vs_gen = summary_dict.pop('real_vs_gen')
    with tf.contrib.summary.create_file_writer(summary_dir, max_queue=summary_queue_len).as_default():
        with tf.contrib.summary.record_summaries_every_n_global_steps(image_summary_freq, global_step=gs):
            tf.contrib.summary.image("real_vs_gen", real_vs_gen, step=gs)
        
        with tf.contrib.summary.always_record_summaries():   
            for k, v in summary_dict.items():
                tf.contrib.summary.scalar(k, v, step=gs)
        return tf.contrib.summary.all_summary_ops()


def wrap_host(summary_dir, summary_queue_len, image_summary_freq, fn):
    def fn1(**kwargs):
        return fn(summary_dir, summary_queue_len, image_summary_freq, **kwargs)
    return fn1


class DeterministicModel(BaseModel):
    def _model_default_hparams(self):
        return {
            "lr": 0.001,
            "end_lr": 0.0,
            "decay_steps": (200000, 300000),
            "max_steps": 300000,
            "beta1": 0.9,
            "beta2": 0.999,
            'l1_weight': 1.0,
            'l2_weight': 0.0,
            'num_scales': 1,
            'vgg_cdist_weight': 0.0,
            'state_weight': 0.0,
            'tv_weight': 0.001,
            "tpu_log_pad": 5
        }

    def _model_fn(self, model_inputs, model_targets, mode):
        # prep inputs here
        logger = logging.getLogger(__name__)
        inputs, targets = {}, {}
        inputs['actions'], inputs['images'] = tf.transpose(model_inputs['actions'], [1, 0, 2]), tf.transpose(model_inputs['images'], [1, 0, 2, 3, 4])
        if mode == tf.estimator.ModeKeys.TRAIN:
            targets['images'] = tf.transpose(model_targets['images'][:, self._hparams.context_frames:], [1, 0, 2, 3, 4])
        
        if self._hparams.use_states:
            inputs['states'] = tf.transpose(model_inputs['states'][:, self._hparams.context_frames:], [1, 0, 2])
            if self._hparams.state_weight and mode == tf.estimator.ModeKeys.TRAIN:
                targets['states'] = tf.transpose(model_targets['state'], [1, 0, 2])
            else:
                logger.warning('states supplied but state_weight=0 so no loss will be computed on predicted states')
        elif self._hparams.state_weight > 0:
            raise ValueError("states not supplied but state_weight > 0")
        
        # if annotations are present construct 'pixel flow error metric'
        if 'annotations' in model_inputs or 'pixel_distributions' in model_inputs:
            if mode == tf.estimator.ModeKeys.TRAIN:
                inputs['pix_distribs'] = tf.transpose(model_inputs['annotations'], [1, 0, 2, 3, 4])
                targets['pix_distribs'] = tf.transpose(model_targets['annotations'][:, self._hparams.context_frames:], [1, 0, 2, 3, 4])
            else:
                inputs['pix_distribs'] = tf.transpose(model_inputs['pixel_distributions'], [1, 0, 2, 3, 4])

        if 'encoder' in self._hparams and self._hparams.encoder == 'one_step':
            assert mode == tf.estimator.ModeKeys.TRAIN
            tlen = inputs['images'].get_shape().as_list()[0]
            inputs_tr_inf, targets_tr_inf = split_model_inference(inputs, targets, self._hparams)
            outputs_enc = onestep_encoder_fn(inputs_tr_inf['inference'], self._hparams)
            self._hparams.e_dim = outputs_enc.get_shape().as_list()[2]
            outputs_enc = average_and_repeat(outputs_enc, self._hparams, tlen)
            inputs = inputs_tr_inf['train']
            targets = targets_tr_inf['train']
        else:
            outputs_enc = None
        inputs['outputs_enc'] = outputs_enc

        # build the graph
        model_graph = self._graph_class()

        if self._num_gpus <= 1:
            outputs = model_graph.build_graph(mode, inputs, self._hparams, self._graph_scope)
        else:
            # TODO: add multi-gpu support
            raise NotImplementedError
        pred_frames = tf.transpose(outputs["gen_images"], [1,0,2,3,4])

        # if train build the loss function (don't support multi-gpu training)
        if mode == tf.estimator.ModeKeys.TRAIN:
            assert self._num_gpus <= 1, "only single gpu training supported at the moment"
            global_step = tf.train.get_or_create_global_step()
            lr, optimizer = tf_utils.build_optimizer(self._hparams.lr, self._hparams.beta1, self._hparams.beta2, 
                                        decay_steps=self._hparams.decay_steps, 
                                        end_lr=self._hparams.end_lr,
                                        global_step=global_step)
            if self._tpu_mode and self._use_tpu:
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)


            gen_losses = OrderedDict()
            if not (self._hparams.l1_weight or self._hparams.l2_weight or self._hparams.vgg_cdist_weight):
                logger.error('no image loss is being created!')
                raise ValueError
            
            gen_images = outputs.get('gen_images_enc', outputs['gen_images'])
            target_images = targets['images']

            scalar_summaries = {'learning_rate': lr}
            tensor_summaries = {'pred_frames': pred_frames}

            if 'encoder' in self._hparams and self._hparams.encoder == 'one_step':
                tensor_summaries['inference_images'] = inputs_tr_inf['inference']['images']
                tensor_summaries['pred_targets'] = target_images
                tensor_summaries['pred_target_dists'] = targets_tr_inf['train']['pix_distribs']

            if 'annotations' in model_inputs:
                tensor_summaries['pred_distrib'] = tf.transpose(outputs['gen_pix_distribs'], [1, 0, 2, 3, 4])
                expected_dist = metrics.expected_pixel_distance(targets['pix_distribs'], outputs['gen_pix_distribs'])
                expected_square_dist = metrics.expected_square_pixel_distance(targets['pix_distribs'], outputs['gen_pix_distribs'])
                var_dist = expected_square_dist - tf.square(expected_dist)
                expected_dist, var_dist = [tf.reduce_sum(x, 0) for x in [expected_dist, var_dist]]

                scalar_summaries['robot_pixel_distance'] = tf.reduce_mean(expected_dist[:, 0])
                scalar_summaries['robot_pixel_var'] = tf.reduce_mean(var_dist[:, 0])
                if expected_dist.get_shape().as_list()[-1] > 1:
                    for o in range(1, expected_dist.get_shape().as_list()[-1]):
                        scalar_summaries['object{}_pixel_distance'.format(o)] = tf.reduce_mean(expected_dist[:, o])
                        scalar_summaries['object{}_pixel_var'.format(o)] = tf.reduce_mean(var_dist[:, o])
        
            if 'ground_truth_sampling_mean' in outputs:
                scalar_summaries['ground_truth_sampling_mean'] = outputs['ground_truth_sampling_mean']
            
            if self._hparams.l1_weight:
                gen_l1_loss = losses.l1_loss(gen_images, target_images)
                gen_losses["gen_l1_loss"] = (gen_l1_loss, self._hparams.l1_weight)
                scalar_summaries['l1_loss'] = gen_l1_loss
            
            if self._hparams.l2_weight:
                gen_l2_loss = losses.l2_loss(gen_images, target_images)
                gen_losses["gen_l2_loss"] = (gen_l2_loss, self._hparams.l2_weight)
                scalar_summaries['l2_loss'] = gen_l2_loss

            if (self._hparams.l1_weight or self._hparams.l2_weight) and self._hparams.num_scales > 1:
                for i in range(1, self._hparams.num_scales):
                    scale_factor = 2 ** i
                    gen_images_scale = tf_utils.with_flat_batch(pool2d)(gen_images, scale_factor, scale_factor, pool_mode='avg')
                    target_images_scale = tf_utils.with_flat_batch(pool2d)(target_images, scale_factor, scale_factor, pool_mode='avg')
                    if self._hparams.l1_weight:
                        gen_l1_scale_loss = losses.l1_loss(gen_images_scale, target_images_scale)
                        gen_losses["gen_l1_scale%d_loss" % i] = (gen_l1_scale_loss, self._hparams.l1_weight)
                        scalar_summaries['l1_loss_scale{}'.format(i)] = gen_l1_scale_loss

                    if self._hparams.l2_weight:
                        gen_l2_scale_loss = losses.l2_loss(gen_images_scale, target_images_scale)
                        gen_losses["gen_l2_scale%d_loss" % i] = (gen_l2_scale_loss, self._hparams.l2_weight)
                        scalar_summaries['l2_loss_scale{}'.format(i)] = gen_l2_scale_loss
                
            if self._hparams.vgg_cdist_weight:
                gen_vgg_cdist_loss = metrics.vgg_cosine_distance(gen_images, target_images)
                gen_losses['gen_vgg_cdist_loss'] = (gen_vgg_cdist_loss, self._hparams.vgg_cdist_weight)
                scalar_summaries['vgg_cdist_loss'] = gen_vgg_cdist_loss

            if self._hparams.state_weight:
                gen_states = outputs.get('gen_states_enc', outputs['gen_states'])
                target_states = targets['states']
                gen_state_loss = losses.l2_loss(gen_states, target_states)
                gen_losses["gen_state_loss"] = (gen_state_loss, self._hparams.state_weight)
                metric_summaries['state_loss'] = gen_state_loss

            if self._hparams.tv_weight:
                gen_flows = outputs.get('gen_flows_enc', outputs['gen_flows'])
                flow_diff1 = gen_flows[..., 1:, :, :, :] - gen_flows[..., :-1, :, :, :]
                flow_diff2 = gen_flows[..., :, 1:, :, :] - gen_flows[..., :, :-1, :, :]
                # sum over the multiple transformations but take the mean for the other dimensions
                gen_tv_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(flow_diff1), axis=(-2, -1))) + \
                            tf.reduce_mean(tf.reduce_sum(tf.abs(flow_diff2), axis=(-2, -1)))
                gen_losses['gen_tv_loss'] = (gen_tv_loss, self._hparams.tv_weight)
                scalar_summaries['tv_loss'] = gen_tv_loss

            loss = sum(loss * weight for loss, weight in gen_losses.values())

            print('computing gradient and train_op')
            g_gradvars = optimizer.compute_gradients(loss, var_list=model_graph.vars)
            g_train_op = optimizer.apply_gradients(g_gradvars, global_step=global_step)
            
            if self._tpu_mode:
                import numpy as np
                try:
                    parameter_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
                    print("parameter_count =", parameter_count)
                except TypeError:
                    pass

                log_summaries = {}
                log_summaries['global_step'] = tf.reshape(global_step, [1])
                for k in scalar_summaries.keys():
                    log_summaries[k]= tf.reshape(scalar_summaries[k], [1])
                
                reals, gen = [tf.split(tf.transpose(tens, [1, 0, 2, 3, 4]), tens.get_shape().as_list()[1], axis=0) for tens in [target_images, gen_images]]
                reals, gen = [[tf.concat(tf.split(i[0], i.get_shape().as_list()[1], axis=0), axis=-2)[0]  for i in img]  for img in (reals, gen)]
                pad = tf.ones([self._hparams.tpu_log_pad] + reals[0].get_shape().as_list()[1:])
                real_gen = [tf.concat((r, pad, g), axis=0)  for r, g in zip(reals, gen)]
                
                log_tensor = [real_gen[0]]
                for rg in real_gen[1:]:
                    log_tensor.extend([pad, pad, rg])
                log_tensor = tf.concat(log_tensor, axis=0)[None]

                log_summaries['real_vs_gen'] = tf.clip_by_value(log_tensor, 0, 1)
                host_fn = wrap_host(self._summary_dir, self._summary_queue_len, self._image_summary_freq, host_summary_fn)
                return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, train_op=g_train_op, host_call=(host_fn, log_summaries))
            
            est = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=g_train_op)
            return est, scalar_summaries, tensor_summaries

        ret_dict = {'predicted_frames': pred_frames[:, :, None]}
        if 'gen_pix_distribs' in outputs:
            ret_dict['predicted_pixel_distributions'] = tf.transpose(outputs['gen_pix_distribs'], [1, 0, 2, 3, 4])[:, :, None]
        return ret_dict
