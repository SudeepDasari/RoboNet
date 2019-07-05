from robonet.datasets.robonet_dataset import RoboNetDataset
from robonet.datasets.util.dataset_utils import split_train_val_test


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
        return parent_hparams

    def _split_files(self, source_number, metadata):
        assert self._hparams.load_annotations, "mode requires annotation loading"
        assert self._hparams.zero_if_missing_annotation, "mode requires some files to not be annotated"

        non_annotated_metadata = metadata[metadata['contains_annotation'] != True]
        
        if self._hparams.train_ex_per_source != [-1]:
            train_files, val_files, test_files = split_train_val_test(metadata, train_ex=self._hparams.train_ex_per_source[source_number], rng=self._random_generator['base'])
        else:
            train_files, val_files, test_files = split_train_val_test(non_annotated_metadata, splits=self._hparams.splits, rng=self._random_generator['base'])
        
        all_annotated = metadata[metadata['contains_annotation'] == True]
        robot_files = [all_annotated[all_annotated['robot'] == r].files for r in self._annotated_robots]

        if len(self._annotated_robots) == 1:
            return [train_files, val_files, test_files] + robot_files
        return [train_files, val_files, test_files] + [all_annotated.files] + robot_files
    
    @property
    def modes(self):
        if self._annotated_robots is None:
            self._annotated_robots = []
            for m in self._metadata:
                annotated_robots_from_source = m[m['contains_annotation'] == True]['robot'].frame.unique().tolist()
                self._annotated_robots.extend(annotated_robots_from_source)
            self._annotated_robots = list(set(self._annotated_robots))

        all_annotated_mode = []
        if len(self._annotated_robots) > 1:
            all_annotated_mode = ['all_annotated']
    
        return ['train', 'val', 'test'] + all_annotated_mode + ['{}_annotated'.format(r) for r in self._annotated_robots]


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

    hparams = {'ret_fnames': True, 'load_T': args.load_steps,'action_mismatch': 3, 'state_mismatch': 3, 'splits':[0.8, 0.1, 0.1], 'same_cam_across_sub_batch':False}
    loader = AnnotationBenchmarkDataset(args.batch_size, args.path, hparams=hparams)
    print('modes are', loader.modes)

    tensors = [loader[x, args.mode] for x in ['images', 'states', 'actions', 'annotations', 'f_names']]
    s = tf.Session()
    out_tensors = s.run(tensors, feed_dict=loader.build_feed_dict(args.mode))
    
    import imageio
    writer = imageio.get_writer('test_frames.gif')
    for t in range(out_tensors[0].shape[1]):
        writer.append_data((np.concatenate([b for b in out_tensors[0][:, t, 0]], axis=-2) * 255).astype(np.uint8))
    writer.close()
    import pdb; pdb.set_trace()
    print('loaded tensors!')
