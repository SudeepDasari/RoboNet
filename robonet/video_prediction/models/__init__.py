from .graphs import get_graph_class


def get_model(class_name):
    if class_name == 'deterministic':
        from .deterministic_generator import DeterministicModel
        return DeterministicModel
    else:
        raise NotImplementedError
