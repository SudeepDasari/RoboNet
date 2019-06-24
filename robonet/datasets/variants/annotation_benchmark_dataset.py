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
        parent_hparams.load_annotations = True
        parent_hparams.zero_if_missing_annotation = True
        return parent_hparams
    
    def _init_train_val_test_files(self):
        min_steps = int(min(min(self._metadata.frame['img_T']), min(self._metadata.frame['state_T'])))
        if not self._hparams.load_T:
            self._hparams.load_T = min_steps
        else:
            assert self._hparams.load_T <= min_steps, "ask to load {} steps but some records only have {}!".format(self._hparams.min_T, min_steps)
        
        assert self._hparams.splits[1], "mode only works with validation records"
        assert self._hparams.load_annotations, "mode requires annotation loading"
        assert self._hparams.zero_if_missing_annotation, "mode requires some files to not be annotated"

        train_test_files = self._metadata[self._metadata['contains_annotation'] != True].files
        val_files = self._metadata[self._metadata['contains_annotation'] == True].files
        [self.rng.shuffle(files) for files in [train_test_files, val_files]]
        train_pivot = int(len(train_test_files) * self._hparams.splits[0])

        if self._hparams.splits[0]:
            self._train_files = train_test_files[:train_pivot]
        if self._hparams.splits[1]:
            assert len(val_files), "no files have annotations!"
            self._val_files = val_files
        if self._hparams.splits[2]:
            self._test_files = train_test_files[train_pivot:]
