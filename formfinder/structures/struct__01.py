# FD basic example, a line


def generate_struct():
    # Example usage
    nodes = {
        1: (0.0, 0.0, 0.0),
        2: (10.0, 0.0, 0.0),
    }

    elements = {
        1: (1, 2),
    }

    elements_preload = {
        1: 0.1,
    }

    nodes_loads = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
    }

    nodes_fixed = {
        1: 1,
        2: 0,
    }

    return nodes, elements, elements_preload, nodes_loads, nodes_fixed
