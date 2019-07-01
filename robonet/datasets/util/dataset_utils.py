
import pdb
import tensorflow as tf

def color_augment(image):
    bs = image.get_shape().as_list()[0]
    shape = [bs] + [1 for _ in range(len(image.get_shape().as_list()) - 1)]
    min = -0.2
    max = 0.2
    rand_h = tf.random_uniform(shape, minval=min, maxval=max)
    rand_s = tf.random_uniform(shape, minval=min, maxval=max)
    rand_v = tf.random_uniform(shape, minval=min, maxval=max)
    image_hsv = tf.image.rgb_to_hsv(image)
    h_, s_, v_ = tf.split(image_hsv, 3, -1)
    stack_mod = tf.clip_by_value(tf.concat([h_ + rand_h, s_ + rand_s, v_ + rand_v], axis=-1), 0, 1.)
    image_rgb = tf.image.hsv_to_rgb(stack_mod)
    return image_rgb
