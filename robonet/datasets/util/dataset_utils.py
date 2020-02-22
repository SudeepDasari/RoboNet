import numpy as np


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
    
    return dict(train=train_files, val=val_files, test=test_files)
