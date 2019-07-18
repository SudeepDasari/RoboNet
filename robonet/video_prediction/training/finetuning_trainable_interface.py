
import glob
import pdb

from robonet.video_prediction.training.trainable_interface import VPredTrainable
from robonet.datasets import get_dataset_class, load_metadata

class BatchmixingVPredTrainable(VPredTrainable):

    def _default_hparams(self):
        params = super()._default_hparams()
        params.add_hparam('robot_set', ['sawyer', 'widowx', 'R3', 'franka'])
        return params

    def make_dataloaders(self,  config):
        DatasetClass = get_dataset_class(self.dataset_hparams.pop('dataset'))

        # data from new domain
        new_domain_metadata = self._filter_metadata(load_metadata(config['data_directory']))

        # data from old domain
        old_domain_metadata = self._filter_metadata(load_metadata(config['batchmix_basedata']))

        old_metadata_list = []
        for m in old_domain_metadata:
            if m['robot'].frame.unique().tolist()[0] in self._hparams.robot_set:
                print('using robot', m['robot'].frame.unique().tolist())
                old_metadata_list.append(m)

        assert len(new_domain_metadata) == 1
        metadata_list = new_domain_metadata*len(old_metadata_list) + old_metadata_list # make sure that we're using the same amount of data from old and new

        return self._get_input_targets(DatasetClass, metadata_list, self.dataset_hparams)

