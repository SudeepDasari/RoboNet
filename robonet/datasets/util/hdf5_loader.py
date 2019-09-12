import h5py
import cv2
import pdb
import imageio
import io
import hashlib
import numpy as np
import os
import random


class ACTION_MISMATCH:
    ERROR = 0
    PAD_ZERO = 1
    CLEAVE = 2


class STATE_MISMATCH:
    ERROR = 0
    PAD_ZERO = 1
    CLEAVE = 2


def default_loader_hparams():
    return {
            'target_adim': 4,
            'target_sdim': 5,
            'state_mismatch': STATE_MISMATCH.ERROR,     # TODO make better flag parsing
            'action_mismatch': ACTION_MISMATCH.ERROR,   # TODO make better flag parsing
            'img_size': [48, 64],
            'cams_to_load': [0],
            'impute_autograsp_action': True,
            'load_annotations': False,
            'zero_if_missing_annotation': False, 
            'load_T': 0                               # TODO implement error checking here for jagged reading
            }


def load_camera_imgs(cam_index, file_pointer, file_metadata, target_dims, start_time=0, n_load=None):
    cam_group = file_pointer['env']['cam{}_video'.format(cam_index)]
    old_dims = file_metadata['frame_dim']
    length = file_metadata['img_T']
    encoding = file_metadata['img_encoding']
    image_format = file_metadata['image_format']

    if n_load is None:
        n_load = length

    old_height, old_width = old_dims
    target_height, target_width = target_dims
    resize_method = cv2.INTER_CUBIC
    if target_height * target_width < old_height * old_width:
        resize_method = cv2.INTER_AREA
    
    images = np.zeros((n_load, target_height, target_width, 3), dtype=np.uint8)
    if encoding == 'mp4':
        buf = io.BytesIO(cam_group['frames'][:].tostring())
        img_buffer = [img for t, img in enumerate(imageio.get_reader(buf, format='mp4')) if start_time <= t < n_load + start_time]
    elif encoding == 'jpg':
        img_buffer = [cv2.imdecode(cam_group['frame{}'.format(t)][:], cv2.IMREAD_COLOR)[:, :, ::-1] 
                                for t in range(start_time, start_time + n_load)]
    else: 
        raise ValueError("encoding not supported")
    
    for t, img in enumerate(img_buffer):
        if (old_height, old_width) == (target_height, target_width):
            images[t] = img
        else:
            images[t] = cv2.resize(img, (target_width, target_height), interpolation=resize_method)
    
    if image_format == 'RGB':
        return images
    elif image_format == 'BGR':
        return images[:, :, :, ::-1]
    raise NotImplementedError


def load_states(file_pointer, meta_data, hparams):
    s_T, sdim = meta_data['state_T'], meta_data['sdim']
    if hparams.target_sdim == sdim:
        return file_pointer['env']['state'][:]

    elif sdim < hparams.target_sdim and hparams.state_mismatch & STATE_MISMATCH.PAD_ZERO:
        pad = np.zeros((s_T, hparams.target_sdim - sdim), dtype=np.float32)
        return np.concatenate((file_pointer['env']['state'][:], pad), axis=-1)

    elif sdim > hparams.target_sdim and hparams.state_mismatch & STATE_MISMATCH.CLEAVE:
        return file_pointer['env']['state'][:][:, :hparams.target_sdim]

    else:
        raise ValueError("file sdim - {}, target sdim - {}, pad behavior - {}".format(sdim, hparams.target_sdim, hparams.state_mismatch))


def load_actions(file_pointer, meta_data, hparams):
    a_T, adim = meta_data['action_T'], meta_data['adim']
    if hparams.target_adim == adim:
        return file_pointer['policy']['actions'][:]

    elif hparams.target_adim == adim + 1 and hparams.impute_autograsp_action and meta_data['primitives'] == 'autograsp':
        action_append, old_actions = np.zeros((a_T, 1)), file_pointer['policy']['actions'][:]
        next_state = file_pointer['env']['state'][:][1:, -1]
        
        high_val, low_val = meta_data['high_bound'][-1], meta_data['low_bound'][-1]
        midpoint = (high_val + low_val) / 2.0

        for t, s in enumerate(next_state):
            if s > midpoint:
                action_append[t, 0] = high_val
            else:
                action_append[t, 0] = low_val
        return np.concatenate((old_actions, action_append), axis=-1)

    elif adim < hparams.target_adim and hparams.action_mismatch & ACTION_MISMATCH.PAD_ZERO:
        pad = np.zeros((a_T, hparams.target_adim - adim), dtype=np.float32)
        return np.concatenate((file_pointer['policy']['actions'][:], pad), axis=-1)

    elif adim > hparams.target_adim and hparams.action_mismatch & ACTION_MISMATCH.CLEAVE:
        return file_pointer['policy']['actions'][:][:, :hparams.target_adim]

    else:
        raise ValueError("file adim - {}, target adim - {}, pad behavior - {}".format(adim, hparams.target_adim, hparams.action_mismatch))


def load_annotations(file_pointer, metadata, hparams, cams_to_load):
    old_height, old_width = metadata['frame_dim']
    target_height, target_width = hparams.img_size
    scale_height, scale_width = target_height / float(old_height), target_width / float(old_width)
    annot = np.zeros((metadata['img_T'], len(cams_to_load), target_height, target_width, 2), dtype=np.float32)
    if metadata.get('contains_annotation', False) != True and hparams.zero_if_missing_annotation:
        return annot

    assert metadata['contains_annotation'], "no annotations to load!"
    point_mat = file_pointer['env']['bbox_annotations'][:].astype(np.int32)

    for t in range(metadata['img_T']):
        for n, chosen_cam in enumerate(cams_to_load):
            for obj in range(point_mat.shape[2]):
                h1, w1 = point_mat[t, chosen_cam, obj, 0] * [scale_height, scale_width] - 1
                h2, w2 = point_mat[t, chosen_cam, obj, 1] * [scale_height, scale_width] - 1
                h, w = int((h1 + h2) / 2), int((w1 + w2) / 2)
                annot[t, n, h, w, obj] = 1
    return annot


def load_data(f_name, file_metadata, hparams, rng=None):
    rng = random.Random(rng)

    assert os.path.exists(f_name) and os.path.isfile(f_name), "invalid f_name"
    with open(f_name, 'rb') as f:
        buf = f.read()
    assert hashlib.sha256(buf).hexdigest() == file_metadata['sha256'], "file hash doesn't match meta-data. maybe delete pkl and re-generate?"
    
    with h5py.File(io.BytesIO(buf)) as hf:
        start_time, n_states = 0, min([file_metadata['state_T'], file_metadata['img_T'], file_metadata['action_T'] + 1])
        assert n_states > 1, "must be more than one state in loaded tensor!"
        if 1 < hparams.load_T < n_states:
            start_time = rng.randint(0, n_states - hparams.load_T)
            n_states = hparams.load_T

        assert all([0 <= i < file_metadata['ncam'] for i in hparams.cams_to_load]), "cams_to_load out of bounds!"
        images, selected_cams = [], []
        for cam_index in hparams.cams_to_load:
            images.append(load_camera_imgs(cam_index, hf, file_metadata, hparams.img_size, start_time, n_states)[None])
            selected_cams.append(cam_index)
        images = np.swapaxes(np.concatenate(images, 0), 0, 1)

        actions = load_actions(hf, file_metadata, hparams).astype(np.float32)[start_time:start_time + n_states-1]
        states = load_states(hf, file_metadata, hparams).astype(np.float32)[start_time:start_time + n_states]

        if hparams.load_annotations:
            annotations = load_annotations(hf, file_metadata, hparams, selected_cams)[start_time:start_time + n_states]
            return images, actions, states, annotations

    return images, actions, states


if __name__ == '__main__':
    import argparse
    import tensorflow as tf
    import robonet.datasets as datasets
    import random
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="tests hdf5 data loader without tensorflow dataset wrapper")
    parser.add_argument('file', type=str, help="path to hdf5 you want to load")
    parser.add_argument('--load_annotations', action='store_true', help="loads annotations if supplied")
    parser.add_argument('--load_steps', type=int, default=0, help="loads <load_steps> steps from the dataset instead of everything")
    args = parser.parse_args()
    
    assert 'hdf5' in args.file
    data_folder = '/'.join(args.file.split('/')[:-1])
    meta_data = datasets.load_metadata(data_folder)

    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.load_T = args.load_steps
    if args.load_annotations:
        hparams.load_annotations = True
        print(meta_data[meta_data['contains_annotation'] == True])
        meta_data = meta_data[meta_data['contains_annotation'] == True]
        imgs, actions, states, annot = load_data((args.file, meta_data.get_file_metadata(args.file)), hparams)
    else:
        imgs, actions, states = load_data((args.file, meta_data.get_file_metadata(args.file)), hparams)
    
    print('actions', actions.shape)
    print('states', states.shape)
    print('images', imgs.shape)
    
    if args.load_annotations:
        for o in range(2):
            w = imageio.get_writer('out{}.gif'.format(o))
            for t, i in enumerate(imgs):
                dist_render = plt.cm.viridis(annot[t, :, :, o])[:, :, :3]
                w.append_data((i * dist_render).astype(np.uint8))
            w.close()
    else:
        w = imageio.get_writer('out.gif')
        for i in imgs:
            w.append_data(i)
        w.close()

