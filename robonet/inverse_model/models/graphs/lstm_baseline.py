from robonet.inverse_model.models.graphs.base_graph import BaseGraph
import itertools
import tensorflow as tf
import tensorflow.keras.layers as layers
from robonet.inverse_model.models.layers.vgg_pretrain import get_vgg_dict, vgg_preprocess_images, vgg_conv, vgg_pool


class ImageEncoder(tf.Module):
    def __init__(self, spec, kernel_size, out_dim, vgg_path, padding='same', name=None):
        super(ImageEncoder, self).__init__(name=name)
        self._convs = []
        self._vgg_dict = get_vgg_dict(vgg_path)

        for s in spec:
            if s == 'pool':
                self._convs.append(layers.MaxPool2D())
            else:
                self._convs.append(layers.Conv2D(s, kernel_size, padding=padding))
        
        self._top = layers.Dense(out_dim)
 
    def __call__(self, input_img):
        preprocessed = vgg_preprocess_images(input_img)
        conv1_out = vgg_conv(self._vgg_dict, vgg_conv(self._vgg_dict, preprocessed, "conv1_1"), "conv1_2")
        pool_out = vgg_pool(conv1_out, "pool1")

        conv2_out = vgg_conv(self._vgg_dict, vgg_conv(self._vgg_dict, conv1_out, "conv2_1"), "conv2_2")
        pool_out = vgg_pool(conv2_out, "pool2")

        top = pool_out
        for c in self._convs:
            top = tf.nn.relu(c(top))
        B = input_img.get_shape().as_list()[0]

        return tf.nn.relu(self._top(tf.reshape(top, (B, -1))))
 

class LSTMBaseline(BaseGraph):
    def build_graph(self, mode, inputs, hparams, scope_name='flow_generator'):
        self._scope_name = scope_name
        outputs = {}
        with tf.variable_scope(scope_name) as graph_scope:
            encoder = ImageEncoder(hparams.spec, hparams.kernel_size, hparams.enc_dim, hparams.vgg_path)
            import pdb; pdb.set_trace()
            start_enc = encoder(inputs['start_images'])
            goal_enc = encoder(inputs['goal_images'])
            start_goal_enc = tf.concat((start_enc, goal_enc), -1)

            lstm_in = layers.Dense(hparams.latent_dim * inputs['T'])(start_goal_enc)
            lstm_in = tf.reshape(lstm_in, (-1, inputs['T'], hparams.latent_dim))
            lstm_out = layers.LSTM(hparams.lstm_dim, return_sequences=True)(lstm_in)

            outputs['pred_actions'] = layers.Dense(inputs['adim'])(lstm_out)

            return outputs

    @staticmethod
    def default_hparams():
        default_params =  {
            "spec": [256, 256, 'pool', 256, 256, 'pool'],
            "enc_dim": 100,
            "kernel_size": 3,

            "latent_dim": 20,
            "lstm_dim": 128,

            "vgg_path": '~/'
        }
        return dict(itertools.chain(BaseGraph.default_hparams().items(), default_params.items()))
