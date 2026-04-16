# -----------------------------------------------------------------------------
# Example input file
# Validation structure based on:
# "An outline of the natural force density method and its extension to
# quadrilateral elements"
# International Journal of Solids and Structures
# -----------------------------------------------------------------------------


def generate_struct(N=11, spacing=1.0, ratio_outer_to_inner=30):
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
    # N x N square grid with uniform spacing.
    # Corner nodes are assigned prescribed z-values.
    # All other edge and interior nodes start in the z = 0 plane.
    # Node numbering proceeds row by row:
    # left to right, then bottom to top.
    # -------------------------------------------------------------------------
    corner_z = {
        (0, 0): -1.5,
        (0, N - 1): 1.5,
        (N - 1, 0): 1.5,
        (N - 1, N - 1): -1.5,
    }

    node_id = 1
    for i in range(N):
        for j in range(N):
            x = i * spacing
            y = j * spacing

            if (i, j) in corner_z:
                z = corner_z[(i, j)]
            else:
                z = 0.0

            nodes[node_id] = (x, y, z)
            node_id += 1

    # -------------------------------------------------------------------------
    # Boundary conditions
    # 0 = free
    # 1 = fixed
    # Only the four corner nodes are fixed.
    # -------------------------------------------------------------------------
    corner_ids = [1, N, N * (N - 1) + 1, N * N]

    for node_id in nodes:
        if node_id in corner_ids:
            nodes_fixed[node_id] = 1
        else:
            nodes_fixed[node_id] = 0

    # -------------------------------------------------------------------------
    # Applied nodal loads
    # Format: {node_id: (Fx, Fy, Fz)}
    # No external nodal loads are applied by default.
    # -------------------------------------------------------------------------
    nodes_loads = {node_id: (0.0, 0.0, 0.0) for node_id in nodes}

    # Example optional nodal loads:
    # nodes_loads[12] = (0.0, 0.0, 5.0)
    # nodes_loads[15] = (0.0, 5.0, 0.0)

    # -------------------------------------------------------------------------
    # Element connectivity
    # Format: {element_id: (start_node, end_node)}
    # Elements connect the orthogonal grid in the x and y directions.
    # -------------------------------------------------------------------------
    element_id = 1
    for i in range(N):
        for j in range(N):
            current_node = i * N + j + 1

            if i + 1 < N:
                right_node = (i + 1) * N + j + 1
                elements[element_id] = (current_node, right_node)
                element_id += 1

            if j + 1 < N:
                top_node = i * N + (j + 1) + 1
                elements[element_id] = (current_node, top_node)
                element_id += 1

    # -------------------------------------------------------------------------
    # Initial element preload / force-density values
    # In FD-based methods, this is the prescribed q or s-type value.
    # Boundary elements are assigned a larger value relative to interior membs
    # according to ratio_outer_to_inner.
    # -------------------------------------------------------------------------
    edge_elements = identify_edge_elements(elements, N)
    s_values = generate_preload(elements, edge_elements, ratio_outer_to_inner)

    for element_id, s_value in zip(elements.keys(), s_values):
        elements_preload[element_id] = float(s_value)

    return nodes, elements, elements_preload, nodes_loads, nodes_fixed


def identify_edge_elements(elements, N):
    # -------------------------------------------------------------------------
    # Identify perimeter elements
    # A member is treated as a boundary element if both of its end nodes lie
    # on the outer perimeter of the N x N grid.
    # -------------------------------------------------------------------------
    boundary_nodes = set()

    for node_id in range(1, N * N + 1):
        i = (node_id - 1) // N
        j = (node_id - 1) % N

        if i == 0 or j == 0 or i == N - 1 or j == N - 1:
            boundary_nodes.add(node_id)

    edge_elements = []
    for element_id, (n1, n2) in elements.items():
        if n1 in boundary_nodes and n2 in boundary_nodes:
            edge_elements.append(element_id)

    return edge_elements


def generate_preload(elements, edge_elements, ratio_outer_to_inner=1):
    # -------------------------------------------------------------------------
    # Assign preload values to each element
    # Interior members are assigned 1.0
    # Boundary members are assigned ratio_outer_to_inner
    # -------------------------------------------------------------------------
    s_values = [1.0] * len(elements)

    for edge_id in edge_elements:
        s_values[edge_id - 1] = float(ratio_outer_to_inner)

    return s_values
