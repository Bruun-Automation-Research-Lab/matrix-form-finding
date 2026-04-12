# -----------------------------------------------------------------------------
# Example input file
# Simple 5-node star example
# Four corner nodes are fixed, and one center node is free
# -----------------------------------------------------------------------------


def generate_struct():
    # -------------------------------------------------------------------------
    # Node coordinates
    # Four corner nodes define a square in the XY plane.
    # One node is placed at the center.
    # -------------------------------------------------------------------------
    nodes = {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        3: (2.0, 2.0, 0.0),
        4: (0.0, 2.0, 0.0),
        5: (1.0, 1.0, 0.0),
    }

    # -------------------------------------------------------------------------
    # Element connectivity
    # Format: {element_id: (start_node, end_node)}
    # Each corner node is connected to the center node.
    # -------------------------------------------------------------------------
    elements = {
        1: (1, 5),
        2: (2, 5),
        3: (3, 5),
        4: (4, 5),
    }

    # -------------------------------------------------------------------------
    # Initial element preload / force-density values
    # In FD_linear, this is the prescribed q value for each element.
    # -------------------------------------------------------------------------
    elements_preload = {
        1: 1.0,
        2: 2.0,
        3: 1.0,
        4: 1.0,
    }

    # -------------------------------------------------------------------------
    # Applied nodal loads
    # Format: {node_id: (Fx, Fy, Fz)}
    # No external loads are applied in this example.
    # -------------------------------------------------------------------------
    nodes_loads = {
        1: (0.0, 0.0, 0.0),
        2: (0.0, 0.0, 0.0),
        3: (0.0, 0.0, 0.0),
        4: (0.0, 0.0, 0.0),
        5: (0.0, 0.0, 0.0),
    }

    # -------------------------------------------------------------------------
    # Boundary conditions
    # 0 = free
    # 1 = fixed
    # The four corner nodes are fixed, and the center node is free.
    # -------------------------------------------------------------------------
    nodes_fixed = {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 0,
    }

    return nodes, elements, elements_preload, nodes_loads, nodes_fixed
