def get_models(class_name):
    if class_name == 'DeterministicInverseModel':
        from .deterministic_inverse_model import DeterministicInverseModel
        return DeterministicInverseModel
    if class_name == 'DiscretizedInverseModel':
        from .discretized_inverse_model import DiscretizedInverseModel
        return DiscretizedInverseModel
    raise NotImplementedError

