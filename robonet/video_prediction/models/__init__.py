from .graphs import get_graph_class


def get_model_fn(class_name):
    if class_name == 'deterministic':
        from .deterministic_generator import vpred_generator
        return vpred_generator
    elif class_name == 'indep_zr':
        from .indep_zr_generator import vpred_generator
        return vpred_generator
    else:
        raise NotImplementedError
