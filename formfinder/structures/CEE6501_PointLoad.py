# -----------------------------------------------------------------------------
# Example input file
# Rectangular grid example with a single downward nodal load
# Custom CEE6501 class example
# Outer-edge nodes are fixed
# -----------------------------------------------------------------------------


def generate_struct(rows=8, cols=10, spacing=1.0):
    # -------------------------------------------------------------------------
    # Initialize dictionaries
    # -------------------------------------------------------------------------
    nodes = {}
    elements = {}
    elements_preload = {}
    nodes_loads = {}
    nodes_fixed = {}

    # -------------------------------------------------------------------------
    # Node coordinates
    # rows x cols rectangular grid in the XY plane with uniform spacing.
    # All nodes start with z = 0.0.
    # Node numbering proceeds row by row:
    # left to right, then bottom to top.
    # -------------------------------------------------------------------------
    node_id = 1
    for i in range(rows):
        for j in range(cols):
            nodes[node_id] = (j * spacing, i * spacing, 0.0)
            node_id += 1

    # -------------------------------------------------------------------------
    # Applied nodal loads
    # Format: {node_id: (Fx, Fy, Fz)}
    # No external loads are applied initially, except for a single point load
    # at node 17.
    # -------------------------------------------------------------------------
    nodes_loads = {node_id: (0.0, 0.0, 0.0) for node_id in nodes}
    nodes_loads[17] = (2.0, 0.0, -10.0)
    nodes_loads[53] = (2.0, 0.0, -10.0)

    # -------------------------------------------------------------------------
    # Boundary conditions
    # 0 = free
    # 1 = fixed
    # All nodes along the outer edge of the rectangular grid are fixed.
    # -------------------------------------------------------------------------
    for node_id in nodes:
        row = (node_id - 1) // cols
        col = (node_id - 1) % cols

        if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
            nodes_fixed[node_id] = 1
        else:
            nodes_fixed[node_id] = 0

    # -------------------------------------------------------------------------
    # Element connectivity
    # Format: {element_id: (start_node, end_node)}
    # Elements connect the orthogonal grid in the x and y directions.
    # -------------------------------------------------------------------------
    element_id = 1
    for i in range(rows):
        for j in range(cols):
            current_node = i * cols + j + 1

            if j < cols - 1:
                right_node = current_node + 1
                elements[element_id] = (current_node, right_node)
                element_id += 1

            if i < rows - 1:
                upper_node = current_node + cols
                elements[element_id] = (current_node, upper_node)
                element_id += 1

    # -------------------------------------------------------------------------
    # Initial element preload / force-density values
    # In FD_linear, this is the prescribed q value for each element.
    # All members are assigned the same initial value in this example.
    # -------------------------------------------------------------------------
    for element_id in elements:
        elements_preload[element_id] = 1.0

    return nodes, elements, elements_preload, nodes_loads, nodes_fixed
