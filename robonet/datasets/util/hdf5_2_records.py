"""
Converts data from hdf5 format to TFRecord format
"""

import tensorflow as tf
from robonet.datasets.util.hdf5_loader import load_data, default_loader_hparams
from tqdm import tqdm


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_record(filename, trajs):
    writer = tf.python_io.TFRecordWriter(filename)
    for traj in tqdm(trajs):
        images, actions, states = traj
        feature = {}
        feature['images'] = bytes_feature(images.flatten().tostring())
        feature['actions'] = float_feature(actions.flatten().tolist())
        feature['states'] = float_feature(states.flatten().tolist())
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


def _load_hdf5(inputs):
    if len(inputs) == 3:
        f_name, file_metadata, hparams = inputs
        return load_data(f_name, file_metadata, hparams)
    elif len(inputs) == 4:
        f_name, file_metadata, hparams, rng = inputs
        return load_data(f_name, file_metadata, hparams, rng)
    raise ValueError


if __name__ == '__main__':
    import argparse
    from robonet.datasets import load_metadata
    from tensorflow.contrib.training import HParams
    import multiprocessing
    import json
    import copy
    import random
    import os


    parser = argparse.ArgumentParser(description="converts data into tfrecord format for fast TPU loading")
    parser.add_argument('path', type=str, default='./', help='path to input file archive')
    parser.add_argument('--robot', type=str, default='', help='if flag supplied only converts data corresponding to given robot')
    parser.add_argument('--filter_primitive', type=str, default='', help='if flag supplied only converts data with given primitive')
    parser.add_argument('--n_workers', type=int, default=1, help='number of worker threads')
    parser.add_argument('--target_adim', type=int, default=5, help='target action dimension for loading')
    parser.add_argument('--target_sdim', type=int, default=5, help='target state dimension for loading')
    parser.add_argument('--img_dims', type=int, nargs='+', default=[48, 64], help='(height, width) to resize images')
    parser.add_argument('--save_dir', type=str, default='./', help='where to save records')
    parser.add_argument('--ex_per_record', type=int, default=96, help='examples per record file')
    args = parser.parse_args()

    name_dir = 'record_names/' + '/'.join(args.save_dir.split('/')[1:])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)
    
    metadata = load_metadata(args.path)
    if args.robot:
        metadata = metadata[metadata['robot']  == args.robot]
    if args.filter_primitive:
        metadata = metadata[metadata['primitives'] == args.filter_primitive]

    ncam = min(metadata['ncam'].frame.unique().tolist())
    print('loaded {} records with robot={} and primitive={}'.format(len(metadata), args.robot, args.filter_primitive))

    hparams = HParams(**default_loader_hparams())
    hparams.target_adim = args.target_adim
    hparams.target_sdim = args.target_sdim
    hparams.action_mismatch = 3
    hparams.state_mismatch = 3
    hparams.cams_to_load = list(range(ncam))
    hparams.load_T = min(min(metadata['state_T']),min(metadata['img_T'])).frame
    assert len(args.img_dims) == 2, "should be (height, width) tuple"
    hparams.img_size = tuple(args.img_dims)

    print('saving images with adim-{}, sdim-{}, img_dims-{}, T-{}'.format(hparams.target_adim, hparams.target_sdim, hparams.img_size, hparams.load_T))

    record_metadata = {'adim': int(hparams.target_adim), 'sdim': int(hparams.target_sdim), 'img_dim': list(hparams.img_size), 'T': int(hparams.load_T) , 'ncam': ncam}
    json.dump(record_metadata, open('{}/format.json'.format(args.save_dir), 'w'))
    json.dump(record_metadata, open('{}/format.json'.format(name_dir), 'w'))
    pool = multiprocessing.Pool(args.n_workers)
    
    all_files = metadata.files
    random.shuffle(all_files)
    f_ind, r_cntr = 0, 0
    f_names = []
    while f_ind < len(all_files):
        f_load = all_files[f_ind:f_ind + args.ex_per_record]
        fm_load = [metadata.get_file_metadata(f) for f in f_load]
        f_hparams = [copy.deepcopy(hparams) for _ in f_load]
        
        loaded_data = pool.map(_load_hdf5, [(f, fm, fh) for f, fm, fh in zip(f_load, fm_load, f_hparams)])
        f_name = '{}/record{}.tfrecord'.format(args.save_dir, r_cntr)
        save_record(f_name, loaded_data)
        print('saved record{}.tfrecord'.format(r_cntr))
        f_names.append(f_name)

        r_cntr += 1
        f_ind += len(loaded_data)

    json.dump(f_names, open('{}/files.json'.format(args.save_dir), 'w'))
    json.dump(f_names, open('{}/files.json'.format(name_dir), 'w'))
