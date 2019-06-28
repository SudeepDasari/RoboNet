from .ray_util.gif_logger import GIFLogger


def get_trainable(class_name):
    if class_name == 'VPredTrainable':
        from .trainable_interface import VPredTrainable
        return VPredTrainable
    if class_name == 'DetEmbedVPredTrainable':
        from .det_embedding_trainable_interface import DetEmbeddingVPredTrainable
        return DetEmbeddingVPredTrainable
    raise NotImplementedError
