def get_trainable(name):
    if name == 'InverseTrainable':
        from .inverse_trainable import InverseTrainable
        return InverseTrainable
    raise NotImplementedError

