import argparse
import os
import glob
from robonet.datasets import load_metadata
import pickle as pkl
import shutil
from hdf5_loader import load_camera_imgs
import h5py
import numpy as np
import tqdm
import cv2
from multiprocessing import Pool, cpu_count


def _swap(args):
    f, m = args
    fp = h5py.File(f, 'a')

    if fp['env'].attrs['cam_encoding'] == 'jpg':
        return
    
    h, w = m['frame_dim']
    all_video = [load_camera_imgs(c, fp, m, (h, w)) for c in range(m['ncam'])]
    for c in range(m['ncam']):
        fp['env'].pop('cam{}_video'.format(c))
        cam_group = fp['env'].create_group("cam{}_video".format(c))
        for t in range(m['img_T']):
            img = all_video[c][t][:,:,::-1]
            data = cam_group.create_dataset("frame{}".format(t), data=cv2.imencode('.jpg', img)[1])
            data.attrs['shape'] = img.shape
            data.attrs['image_format'] = 'BGR'
    fp['env'].attrs['cam_encoding'] = 'jpg'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="swaps image encoding from more space efficient mp4 to faster jpg loader")
    parser.add_argument('hdf5_path', type=str, help="path to input folder containing hdf5 files")
    args = parser.parse_args()

    metadata = load_metadata(args.hdf5_path)
    
    jobs = [(f, metadata.get_file_metadata(f)) for f in metadata.get_shuffled_files()]
    with Pool(max(int(cpu_count() // 2), 1)) as p:
        r = list(tqdm.tqdm(p.imap(_swap, jobs), total=len(jobs)))
