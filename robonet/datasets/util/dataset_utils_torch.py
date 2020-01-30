import torch
import pdb
import numpy as np
import pytorch_colors as colors


def color_augment(image, noise_range=0.2):
    """Add noise in HSV domain to increase color contrast
    Inputs:
        image                   : torch.Tensor((batch, T, n_cams, height, width, channels))
        noise_range (optional)  : float
    
    Outputs:
        image_rgb               : torch.Tensor((batch, T, n_cams, height, width, channels))

    """
    assert noise_range > 0, "noise_range must be positive"

    bs = image.shape[0]
    shape = [bs] + [1 for _ in range(len(image.shape) - 1)]
    min_noise = -noise_range
    max_noise = noise_range

    height, width, _ = image.shape[-3:]

    # Library requires (batch, channels, height, width) format
    image_hsv = colors.rgb_to_hsv(image.view(-1, 3, height, width))
    h_, s_, v_ = torch.unbind(image_hsv, dim=1)

    rand_h = torch.FloatTensor(h_.shape).uniform_(min_noise, max_noise)
    rand_s = torch.FloatTensor(s_.shape).uniform_(min_noise, max_noise)
    rand_v = torch.FloatTensor(v_.shape).uniform_(min_noise, max_noise)

    stack_mod = torch.clamp(
        torch.stack([h_ + rand_h, s_ + rand_s, v_ + rand_v], dim=1), 0, 1.0
    )
    image_rgb = colors.hsv_to_rgb(stack_mod)
    return image_rgb.view(image.shape)


def split_train_val_test(metadata, splits=None, train_ex=None, rng=None):
    assert (splits is None) != (
        train_ex is None
    ), "exactly one of splits or train_ex should be supplied"
    files = metadata.get_shuffled_files(rng)
    train_files, val_files, test_files = None, None, None

    if splits is not None:
        assert (
            len(splits) == 3
        ), "function requires 3 split parameteres ordered (train, val ,test)"
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
        train_files = files[: splits[0]]
    if splits[1]:
        val_files = files[splits[0] : splits[1]]
    if splits[2]:
        test_files = files[splits[1] : splits[2]]

    return train_files, val_files, test_files
