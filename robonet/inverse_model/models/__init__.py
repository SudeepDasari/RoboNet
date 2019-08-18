def get_models(class_name):
    if class_name == 'DeterministicInverseModel':
        from .deterministic_inverse_model import DeterministicInverseModel
        return DeterministicInverseModel
