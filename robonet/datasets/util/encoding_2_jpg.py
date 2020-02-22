import argparse
import os
import glob
from robonet.datasets import load_metadata
import pickle as pkl
import shutil
from hdf5_loader import HDF5Loader
import h5py
import numpy as np
import tqdm
import cv2
from multiprocessing import Pool, cpu_count


def _swap(args):
    f, m = args
    h, w = m['frame_dim']
    loader = HDF5Loader(f, m, {'img_size': [h, w]})
    if loader.hf['env'].attrs['cam_encoding'] == 'jpg':
        return
    
    all_video = [loader.load_video(c) for c in range(m['ncam'])]
    loader.close()

    hf = h5py.File(f, 'a')
    for c in range(m['ncam']):
        hf['env'].pop('cam{}_video'.format(c))
        cam_group = hf['env'].create_group("cam{}_video".format(c))
        for t in range(m['img_T']):
            img = all_video[c][t][:,:,::-1]
            data = cam_group.create_dataset("frame{}".format(t), data=cv2.imencode('.jpg', img)[1])
            data.attrs['shape'] = img.shape
            data.attrs['image_format'] = 'BGR'
    hf['env'].attrs['cam_encoding'] = 'jpg'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="swaps image encoding from more space efficient mp4 to faster jpg loader")
    parser.add_argument('hdf5_path', type=str, help="path to input folder containing hdf5 files")
    args = parser.parse_args()

    metadata = load_metadata(args.hdf5_path)
    
    jobs = [(f, metadata.get_file_metadata(f)) for f in metadata.get_shuffled_files()]
    with Pool(max(int(cpu_count() // 2), 1)) as p:
        r = list(tqdm.tqdm(p.imap(_swap, jobs), total=len(jobs)))

    os.remove('{}/meta_data.pkl'.format(args.hdf5_path))
    meta = load_metadata(args.hdf5_path)
