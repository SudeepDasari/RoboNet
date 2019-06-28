from robonet.datasets.robonet_dataset import RoboNetDataset


class AnnotationBenchmarkDataset(RoboNetDataset):
    """
    Separates files that have annotations and those which don't
        - files with annotations are loaded as validation files
        - all others are loaded as train/test
    """

    @staticmethod
    def _get_default_hparams():
        parent_hparams = RoboNetDataset._get_default_hparams()
        parent_hparams.add_hparam('held_out_robot', '')
        parent_hparams.load_annotations = True
        parent_hparams.zero_if_missing_annotation = True
        return parent_hparams
    
    def _split_files(self, metadata):
        assert self._hparams.splits[1], "mode only works with validation records"
        assert self._hparams.load_annotations, "mode requires annotation loading"
        assert self._hparams.zero_if_missing_annotation, "mode requires some files to not be annotated"
        train_files, test_files, val_files = [], [], []

        if self._hparams.held_out_robot:
            train_test_files = metadata[metadata['robot'] != self._hparams.held_out_robot].files
            val_files = metadata[metadata['robot'] == self._hparams.held_out_robot].files
            [self.rng.shuffle(files) for files in [train_test_files, val_files]]
            train_pivot = int(len(train_test_files) * self._hparams.splits[0])
        else:
            train_test_files = metadata[metadata['contains_annotation'] != True].files
            val_files = metadata[metadata['contains_annotation'] == True].files
            [self.rng.shuffle(files) for files in [train_test_files, val_files]]
            train_pivot = int(len(train_test_files) * self._hparams.splits[0])

        if self._hparams.splits[0]:
            train_files = train_test_files[:train_pivot]
        if self._hparams.splits[1]:
            val_files = val_files
        if self._hparams.splits[2]:
            test_files = train_test_files[train_pivot:]
        
        return train_files, val_files, test_files


if __name__ == '__main__':
    import argparse
    import tensorflow as tf
    import numpy as np

    parser = argparse.ArgumentParser(description="calculates or loads meta_data frame")
    parser.add_argument('path', help='path to files containing hdf5 dataset')
    parser.add_argument('--robots', type=str, nargs='+', default=None, help='will construct a dataset with batches split across given robots')
    parser.add_argument('--held_out', type=str, default='', help='held out robot')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for test loader (should be even for non-time test demo to work)')
    parser.add_argument('--mode', type=str, default='val', help='mode to grab data from')
    parser.add_argument('--load_steps', type=int, default=0, help='if value is provided will load <load_steps> steps')
    args = parser.parse_args()

    hparams = {'ret_fnames': True, 'load_T': args.load_steps, 'sub_batch_size': 2, 'action_mismatch': 3, 'state_mismatch': 3,
              'held_out_robot': args.held_out}
    if args.robots:
        from robonet.datasets import load_metadata
        meta_data = load_metadata(args.path)
        hparams['same_cam_across_sub_batch'] = True
        loader = AnnotationBenchmarkDataset(args.batch_size, [meta_data[meta_data['robot'] == r] for r in args.robots], hparams=hparams)
    else:
        loader = AnnotationBenchmarkDataset(args.batch_size, args.path, hparams=hparams)

    tensors = [loader[x, args.mode] for x in ['images', 'states', 'actions', 'f_names', 'annotations']]
    s = tf.Session()
    out_tensors = s.run(tensors)
    
    import imageio
    writer = imageio.get_writer('test_frames.gif')
    for t in range(out_tensors[0].shape[1]):
        writer.append_data((np.concatenate([b for b in out_tensors[0][:, t, 0]], axis=-2) * 255).astype(np.uint8))
    writer.close()
    import pdb; pdb.set_trace()
    print('loaded tensors!')
