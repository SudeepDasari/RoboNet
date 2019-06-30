from robonet.video_prediction.training.trainable_interface import VPredTrainable
import pdb

class BalancedCamFilter(VPredTrainable):

    def _default_hparams(self):
        params = super()._default_hparams()
        params.add_hparam('balanced_camera_configurations', True)
        return params

    def _filter_metadata(self, metadata):
        metadata = super()._filter_metadata(metadata)

        if self._hparams.balanced_camera_configurations:
            assert self.dataset_hparams.get('sub_batch_size', 1) > 1
            unique_cameras = metadata['camera_configuration'].frame.unique().tolist()   # all camera configs that are in  he dataset
            all_metadata = metadata
            metadata = [all_metadata[all_metadata['camera_configuration'] == r] for r in unique_cameras]

            # print('sizes after splitting metadata in camera configurations')
            # for m, cam in zip(metadata, unique_cameras):
            #     print('cam {} : numfiles {} robots: {}'.format(cam, len(m.files), m['robot'].frame.unique().tolist()))
        return metadata


class RobotSetFilter(VPredTrainable):

    def _default_hparams(self):
        params = super()._default_hparams()
        params.add_hparam('robot_set', ['sawyer', 'franka', 'widowx'])
        params.set_hparam('balance_across_robots', True)
        return params

    def _filter_metadata(self, metadata):
        metadata = super()._filter_metadata(metadata)

        pdb.set_trace()
        if self._hparams.robot_set is not None:
            assert isinstance(metadata, list)
            metadata = [m[m['robot'][0] in self._hparams.robot_set] for m in metadata]

        return metadata
