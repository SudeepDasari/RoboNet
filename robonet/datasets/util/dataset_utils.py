import tensorflow as tf
import numpy as np


def color_augment(image, noise_range=0.2):
    assert noise_range > 0, "noise_range must be positive"
    
    bs = image.get_shape().as_list()[0]
    shape = [bs] + [1 for _ in range(len(image.get_shape().as_list()) - 1)]
    min_noise = -noise_range
    max_noise = noise_range
    rand_h = tf.random_uniform(shape, minval=min_noise, maxval=max_noise)
    rand_s = tf.random_uniform(shape, minval=min_noise, maxval=max_noise)
    rand_v = tf.random_uniform(shape, minval=min_noise, maxval=max_noise)
    image_hsv = tf.image.rgb_to_hsv(image)
    h_, s_, v_ = tf.split(image_hsv, 3, -1)
    stack_mod = tf.clip_by_value(tf.concat([h_ + rand_h, s_ + rand_s, v_ + rand_v], axis=-1), 0, 1.)
    image_rgb = tf.image.hsv_to_rgb(stack_mod)
    return image_rgb


def split_train_val_test(metadata, splits=None, train_ex=None, rng=None):
    assert (splits is None) != (train_ex is None), "exactly one of splits or train_ex should be supplied"
    files = metadata.get_shuffled_files(rng)
    train_files, val_files, test_files = None, None, None

    if splits is not None:
        assert len(splits) == 3, "function requires 3 split parameteres ordered (train, val ,test)"
        splits = np.cumsum([int(i * len(files)) for i in splits]).tolist()
    else:
        assert len(files) >= train_ex, "not enough files for train examples!"
        val_split = int(0.5 * (len(files) + train_ex))
        splits = [train_ex, val_split, len(files)]
    
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
