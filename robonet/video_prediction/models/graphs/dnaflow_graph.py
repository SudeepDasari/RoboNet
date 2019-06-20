from .base_graph import BaseGraph
from robonet.video_prediction.layers.dnaflow_rnn_cell import VPredCell
import itertools
import tensorflow as tf
from robonet.video_prediction.utils import tf_utils


class DNAFlowGraphWrapper(BaseGraph):
    def build_graph(self, inputs, hparams, scope_name='dnaflow_generator'):
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

            'num_domains': 0,
            'zr_dim': 8,
            'za_dim': 4
        }
        return dict(itertools.chain(BaseGraph.default_hparams().items(), default_params.items()))
