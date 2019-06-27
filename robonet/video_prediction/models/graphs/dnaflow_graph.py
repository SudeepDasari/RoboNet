from .base_graph import BaseGraph
from robonet.video_prediction.layers.dnaflow_rnn_cell import VPredCell
import itertools
import tensorflow as tf
from robonet.video_prediction.utils import tf_utils


class DNAFlowGraphWrapper(BaseGraph):
    def build_graph(self, inputs, hparams, outputs_enc=None, scope_name='dnaflow_generator'):
        if hparams.use_states:
            assert "states" in inputs, "graph is building with states but no states in inptus"
        else:
            inputs.pop('states', None)
        
        self._scope_name = scope_name
        with tf.variable_scope(self._scope_name) as graph_scope:
            # TODO: I really don't like this. Should just error at this point instead of padding
            inputs = {name: tf_utils.maybe_pad_or_slice(input, hparams.sequence_length - 1)
                for name, input in inputs.items()}

            if hparams.zr_dim:
                if outputs_enc is None:
                    print('no inference network for zr')
                    zrs_mu = [tf.Variable(tf.zeros([hparams.zr_dim])) for i in range(hparams.num_domains)]
                    zrs_log_sigma = [tf.Variable(tf.zeros([hparams.zr_dim])) for i in range(hparams.num_domains)]
                    zrs = [m + tf.random_normal(tf.shape(m)) * tf.exp(s) for m, s in zip(zrs_mu, zrs_log_sigma)]
                    tiled_zrs = []
                    batch_size = inputs['images'].shape[1].value
                    for i in range(hparams.num_domains):
                        tiled_zr = tf.tile(zrs[i], [(hparams.sequence_length - 1) * batch_size // hparams.num_domains])
                        tiled_zrs.append(tf.reshape(tiled_zr, (hparams.sequence_length - 1, batch_size // hparams.num_domains, hparams.zr_dim)))

                    zrs = tf.concat(tiled_zrs, axis=1)
                    inputs['zrs'] = zrs
                else:
                    batch_size = outputs_enc['enc_zs_mu'].shape[1].value // hparams.num_domains
                    zrs_mu = [outputs_enc['enc_zs_mu'][:, i*batch_size:(i+1)*batch_size] for i in range(hparams.num_domains)]
                    zrs_log_sigma_sq = [outputs_enc['enc_zs_log_sigma_sq'][:, i*batch_size:(i+1)*batch_size] for i in range(hparams.num_domains)]
                    zrs_mu = tf.concat([tf.expand_dims(tf.reshape(m, [-1, hparams.zr_dim]), 0) for m in zrs_mu], axis=0)
                    zrs_log_sigma_sq = tf.concat([tf.expand_dims(tf.reshape(s, [-1, hparams.zr_dim]), 0) for s in zrs_log_sigma_sq], axis=0)

                    def product_of_gaussians(mus, log_sigma_sqs):
                        sigmas = tf.sqrt(tf.exp(log_sigma_sqs))
                        sigmas_squared = tf.square(sigmas)
                        sigmas_squared = tf.clip_by_value(sigmas_squared, 1e-7, 1e7)
                        sigma_squared = 1. / tf.reduce_sum(tf.reciprocal(sigmas_squared), axis=-2)
                        mu = sigma_squared * tf.reduce_sum(mus / sigmas_squared, axis=-2)
                        sigma = tf.sqrt(sigma_squared)
                        return mu, sigma

                    zrs_mu, zrs_sigma = product_of_gaussians(zrs_mu, zrs_log_sigma_sq)
                    eps = tf.random_normal([hparams.num_domains, hparams.zr_dim], 0, 1)
                    zrs = zrs_mu + eps * zrs_sigma
                    tiled_zrs = []
                    for i in range(hparams.num_domains):
                        tiled_zr = tf.tile(zrs[i], [(hparams.sequence_length - 1) * batch_size])
                        tiled_zrs.append(tf.reshape(tiled_zr, (hparams.sequence_length - 1, batch_size, hparams.zr_dim)))

                    zrs = tf.concat(tiled_zrs, axis=1)
                    inputs['zrs'] = zrs
                    inputs['zr_mu'] = tf.tile(tf.expand_dims(zrs_mu, 0), [hparams.sequence_length - 1, batch_size, 1])
                    inputs['zr_log_sigma_sq'] = tf.tile(tf.expand_dims(tf.log(tf.square(zrs_sigma)), 0), [hparams.sequence_length - 1, batch_size, 1])
            
            cell = VPredCell(inputs, hparams)
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32,
                                    swap_memory=False, time_major=True)
        
            outputs = {name: output[hparams.context_frames - 1:] for name, output in outputs.items()}
            outputs['ground_truth_sampling_mean'] = tf.reduce_mean(tf.to_float(cell.ground_truth[hparams.context_frames:]))
        return outputs

    @property
    def vars(self):
        return tf.trainable_variables(self._scope_name)

    @staticmethod
    def default_hparams():
        default_params =  {
            "where_add": "all",
            'ngf': 32,
            'last_frames': 2,
            'num_transformed_images': 4,
            'prev_image_background': True,
            'first_image_background': True,
            'context_images_background': False,
            'generate_scratch_image': False,
            'transformation': "flow",
            'conv_rnn': "lstm",
            'norm_layer': "instance",
            'ablation_conv_rnn_norm': False,
            'downsample_layer': "conv_pool2d",
            'upsample_layer': "upsample_conv2d",
            'dependent_mask': True,
            'c_dna_kernel_size': [5, 5],              # only used in CDNA/DNA mode

            'schedule_sampling': "inverse_sigmoid",
            'schedule_sampling_k': 900.0,
            'schedule_sampling_steps': [0, 100000],
            
            'renormalize_pixdistrib': True,

            'num_domains': 8,
            'zr_dim': 8,
            'za_dim': 4
        }
        return dict(itertools.chain(BaseGraph.default_hparams().items(), default_params.items()))
