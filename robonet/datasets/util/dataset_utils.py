import tensorflow as tf
import numpy as np


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


def split_train_val_test(metadata, splits):
    assert len(splits) == 3, "function requires 3 split parameteres ordered (train, val ,test)"
    files = metadata.files
    train_files, val_files, test_files = None, None, None
    splits = np.cumsum([int(i * len(files)) for i in splits]).tolist()
    
    # give extra fat to val set
    if splits[-1] < len(files):
        diff = len(files) - splits[-1]
        for i in range(1, len(splits)):
            splits[i] += diff
    
    if splits[0]:
        train_files = files[:splits[0]]
    if splits[1]:
        val_files = files[splits[0]: splits[1]]
    if splits[2]:
        test_files = files[splits[1]: splits[2]]
    
    return train_files, val_files, test_files
