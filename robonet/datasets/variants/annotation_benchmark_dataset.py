from robonet.datasets.robonet_dataset import RoboNetDataset


class AnnotationBenchmarkDataset(RoboNetDataset):
    """
    Separates files that have annotations and those which don't
        - files with annotations are loaded as validation files
        - all others are loaded as train/test
    """
    def __init__(self, batch_size, dataset_files_or_metadata, hparams=dict()):
        self._annotated_robots = None
        super(AnnotationBenchmarkDataset, self).__init__(batch_size, dataset_files_or_metadata, hparams)

    @staticmethod
    def _get_default_hparams(parent_hparams=None):
        if parent_hparams is None:
            parent_hparams = RoboNetDataset._get_default_hparams()
        parent_hparams.load_annotations = True
        parent_hparams.zero_if_missing_annotation = True
        parent_hparams.splits = [0.9, 0.1]
        return parent_hparams

    def _split_files(self, metadata):
        assert len(self._hparams.splits) == 2, "mode only support splitting into train/test (val implied by annotations)"
        assert self._hparams.load_annotations, "mode requires annotation loading"
        assert self._hparams.zero_if_missing_annotation, "mode requires some files to not be annotated"

        non_annotated_files = metadata[metadata['contains_annotation'] != True].files
        train_pivot = int(len(non_annotated_files) * self._hparams.splits[0])
        train_files, test_files = non_annotated_files[:train_pivot], non_annotated_files[train_pivot:]
        
        val_metadata = metadata[metadata['contains_annotation'] == True]
        val_files = val_metadata.files
        robot_files = [val_metadata[val_metadata['robot'] == r].files for r in self._annotated_robots]

        return [train_files, val_files, test_files] + robot_files
        
    
    def train_val_filter(self, train_metadata, val_metadata):
        train_metadata = train_metadata[train_metadata['contains_annotation'] != True]
        val_metadata = val_metadata[val_metadata['contains_annotation'] == True]
        print('after filtering annotation files: number of trainfiles {} number of val files {}'.format(len(train_metadata.files), len(val_metadata.files)))
        return train_metadata, val_metadata
    
    @property
    def modes(self):
        if self._annotated_robots is None:
            self._annotated_robots = []
            for m in self._metadata:
                annotated_robots_from_source = m[m['contains_annotation'] == True]['robot'].frame.unique().tolist()
                self._annotated_robots.extend(annotated_robots_from_source)
            self._annotated_robots = list(set(self._annotated_robots))
        return ['train', 'val', 'test'] + ['{}_annotated'.format(r) for r in self._annotated_robots]


if __name__ == '__main__':
    import argparse
    import tensorflow as tf
    import numpy as np
    parser = argparse.ArgumentParser(description="calculates or loads meta_data frame")
    parser.add_argument('path', help='path to files containing hdf5 dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for test loader (should be even for non-time test demo to work)')
    parser.add_argument('--mode', type=str, default='val', help='mode to grab data from')
    parser.add_argument('--load_steps', type=int, default=0, help='if value is provided will load <load_steps> steps')
    args = parser.parse_args()

    hparams = {'ret_fnames': True, 'load_T': args.load_steps,'action_mismatch': 3, 'state_mismatch': 3, 'splits':[0.8, 0.1], 'same_cam_across_sub_batch':True}
    loader = AnnotationBenchmarkDataset(args.batch_size, args.path, hparams=hparams)
    print('modes are', loader.modes)

    tensors = [loader[x, args.mode] for x in ['images', 'states', 'actions', 'annotations', 'f_names']]
    s = tf.Session()
    out_tensors = s.run(tensors)
    
    import imageio
    writer = imageio.get_writer('test_frames.gif')
    for t in range(out_tensors[0].shape[1]):
        writer.append_data((np.concatenate([b for b in out_tensors[0][:, t, 0]], axis=-2) * 255).astype(np.uint8))
    writer.close()
    import pdb; pdb.set_trace()
    print('loaded tensors!')
