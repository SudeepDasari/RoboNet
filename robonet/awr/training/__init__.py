def get_trainable(class_name):
    if class_name == 'AWRTrainable':
        from .awr_trainable import AWRTrainable
        return AWRTrainable
    raise NotImplementedError
