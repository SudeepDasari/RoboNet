import tensorflow as tf
from robonet.video_prediction.ops import lrelu, dense, conv2d, pool2d, get_norm_layer


def create_n_layer_encoder(inputs,
                           nz=8,
                           nef=64,
                           n_layers=3,
                           norm_layer='instance',
                           stochastic=True):
    norm_layer = get_norm_layer(norm_layer)
    layers = []
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

    with tf.variable_scope("layer_1"):
        convolved = conv2d(tf.pad(inputs, paddings), nef, kernel_size=4, strides=2, padding='VALID')
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    for i in range(1, n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = nef * min(2**i, 4)
            convolved = conv2d(tf.pad(layers[-1], paddings), out_channels, kernel_size=4, strides=2, padding='VALID')
            normalized = norm_layer(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    pooled = pool2d(rectified, rectified.shape[1:3].as_list(), padding='VALID', pool_mode='avg')
    squeezed = tf.squeeze(pooled, [1, 2])

    if stochastic:
        with tf.variable_scope('z_mu'):
            z_mu = dense(squeezed, nz)
        with tf.variable_scope('z_log_sigma_sq'):
            z_log_sigma_sq = dense(squeezed, nz)
            z_log_sigma_sq = tf.clip_by_value(z_log_sigma_sq, -10, 10)
        outputs = {'enc_zs_mu': z_mu, 'enc_zs_log_sigma_sq': z_log_sigma_sq}
    else:
        outputs = squeezed
    return outputs