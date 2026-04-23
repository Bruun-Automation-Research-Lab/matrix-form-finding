# -----------------------------------------------------------------------------
# Example input file
# Simple 4-node / 5-element constrained force density example
# Two support nodes are fixed, and two interior nodes are free
# -----------------------------------------------------------------------------

# THIS EXAMPLE MATCHES THE FORCE PATTERN IN THE STRUCTURE CALCULATED
# FROM THE MAIN2.PY CONSTRAINED FD EXAMPLE


def generate_struct():
    # -------------------------------------------------------------------------
    # Node coordinates
    # Two support nodes are fixed along the x-axis.
    # Two interior nodes are placed below them.
    # -------------------------------------------------------------------------
    nodes = {
        1: (0.0, 0.0, 0.0),
        2: (4.0, 0.0, 0.0),
        3: (1.5, -0.5, 0.0),
        4: (2.5, -0.5, 0.0),
    }

    # -------------------------------------------------------------------------
    # Element connectivity
    # Format: {element_id: (start_node, end_node)}
    # -------------------------------------------------------------------------
    elements = {
        1: (1, 3),
        2: (2, 3),
        3: (1, 4),
        4: (2, 4),
        5: (3, 4),
    }

    # -------------------------------------------------------------------------
    # Initial element preload / force-density values
    # In FD_linear, this is the prescribed q value for each element.
    # -------------------------------------------------------------------------

    elements_preload = {
        1: 2.49999999,
        2: 2.29701034,
        3: 1.97386338,
        4: 2.13932387,
        5: 0.17537887,
    }

    # -------------------------------------------------------------------------
    # Applied nodal loads
    # Format: {node_id: (Fx, Fy, Fz)}
    # Downward loads are applied at the two free nodes.
    # -------------------------------------------------------------------------
    nodes_loads = {
        1: (0.0, 0.0, 0.0),
        2: (0.0, 0.0, 0.0),
        3: (0.0, -1.0, 0.0),
        4: (0.0, -1.0, 0.0),
    }

    # -------------------------------------------------------------------------
    # Boundary conditions
    # 0 = free
    # 1 = fixed
    # Nodes 1 and 2 are fixed, nodes 3 and 4 are free.
    # -------------------------------------------------------------------------
    nodes_fixed = {
        1: 1,
        2: 1,
        3: 0,
        4: 0,
    }

    return nodes, elements, elements_preload, nodes_loads, nodes_fixed
