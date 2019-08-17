def get_graph_class(class_name):
    if class_name == 'lstm_baseline':
        from .lstm_baseline import LSTMBaseline
        return LSTMBaseline
    else:
        raise NotImplementedError
