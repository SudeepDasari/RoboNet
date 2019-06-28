from robonet.video_prediction.training.trainable_interface import VPredTrainable
import pdb

class DetEmbeddingVPredTrainable(VPredTrainable):

    def _default_hparams(self):
        params = super()._default_hparams()
        params.add_hparam('balance_camera_configurations', True)
        params.set_hparam('batch_size', 32)
        return params

    def _filter_metadata(self, metadata):
        super()._filter_metadata(metadata)

        if self._hparams.balance_camera_configurations:
            assert self.dataset_hparams.get('sub_batch_size', 1) > 1
            unique_cameras = metadata['camera_configuration'].frame.unique().tolist()   # all camera configs that are in  he dataset
            all_metadata = metadata
            metadata = [all_metadata[all_metadata['camera_configuration'] == r] for r in unique_cameras]

        return metadata

