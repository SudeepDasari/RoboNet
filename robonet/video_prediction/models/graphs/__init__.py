def get_graph_class(class_name):
    if class_name == 'c_dna_flow':
        from .dnaflow_graph import DNAFlowGraphWrapper
        return DNAFlowGraphWrapper
    elif class_name == 'deterministic_graph':
        from .deterministic_graph import DeterministicWrapper
        return DeterministicWrapper
    else:
        raise NotImplementedError
