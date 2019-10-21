import tensorflow as tf
import numpy as np
import os


def get_vgg_dict(path):
    return np.load(os.path.join(path, "vgg19.npy"), encoding='latin1', allow_pickle=True).item()


def vgg_preprocess_images(image_tensor):
    """
    :param image_tensor: float 32 array of Batch x Height x Width x Channel immages (range 0 - 1)
    :return: pre-processed images (ready to input to VGG)
    """
    vgg_mean = tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], dtype=np.float32))
    red, green, blue = tf.split(axis=-1, num_or_size_splits=3, value=image_tensor * 255) 

    return tf.concat(axis=3, values=[
                        blue - vgg_mean[0],
                        green - vgg_mean[1],
                        red - vgg_mean[2],
                     ])


def vgg_conv(vgg_dict, bottom, name):
    with tf.variable_scope(name, reuse=True):
        filt = tf.constant(vgg_dict[name][0], name="filter")

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = tf.constant(vgg_dict[name][1], name="biases")
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu


def vgg_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
