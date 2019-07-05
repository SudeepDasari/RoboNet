from .graphs import get_graph_class


def get_model_fn(class_name):
    if class_name == 'deterministic':
        from .deterministic_generator import vpred_generator
        return vpred_generator
    elif class_name == 'stochastic':
        from .stochastic_generator import vpred_generator
        return vpred_generator
    # elif class_name == 'det_embedding':
    #     from .deterministc_embedding_generator import deterministic_embedding_generator
    #     return deterministic_embedding_generator
    else:
        raise NotImplementedError
