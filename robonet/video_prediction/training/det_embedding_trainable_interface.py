from robonet.video_prediction.training.trainable_interface import VPredTrainable


class DetEmbeddingVPredTrainable(VPredTrainable):

    def _default_hparams(self):
        params = super()._default_hparams()
        params.set_hparam('batch_size', 32)
        return params

