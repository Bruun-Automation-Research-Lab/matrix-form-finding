#  From the paper: "An outline of the natural force density method and its
#  extension to quadrilateral elements"
# International Journal of Solids and Structures

import numpy as np


def generate_struct(N=5, spacing=2.0, ratio_outer_to_inner=20):
    nodes = {}
    elements = {}
    elements_preload = {}
    nodes_load = {}
    nodes_fixed = {}

    # Define corner heights for interpolation
    corner_z = np.array([[-1.5, 1.5], [1.5, -1.5]])

    # Generate nodes (each node as [x, y, z])
    node_id = 1
    for i in range(N):
        for j in range(N):
            x, y = i * spacing, j * spacing

            # Compute z-values based on edge/corner location
            if (i == 0 or i == N - 1) and (j == 0 or j == N - 1):  # Corners
                z = float(corner_z[i // (N - 1), j // (N - 1)])
            else:  # Edge nodes
                z = 0.0  # All edge nodes lie in the z=0 plane

            nodes[node_id] = (x, y, z)
            node_id += 1

    # Define fixed nodes (corners)
    corner_ids = [1, N, N * (N - 1) + 1, N * N]  # IDs of the four corners
    nodes_fixed = {node_id: 1 for node_id in corner_ids}  # 1 for fixed nodes

    # Mark other nodes as free (0)
    for node_id in nodes:
        if node_id not in nodes_fixed:
            nodes_fixed[node_id] = 0

    # Define external loads (zero by default)
    nodes_load = {node_id: (0.0, 0.0, 0.0) for node_id in nodes}

    # Generate elements (grid connectivity)
    element_id = 1
    for i in range(N):
        for j in range(N):
            if i + 1 < N:  # Horizontal connections
                elements[element_id] = ((i * N + j + 1), (i + 1) * N + j + 1)
                element_id += 1
            if j + 1 < N:  # Vertical connections
                elements[element_id] = ((i * N + j + 1), i * N + (j + 1) + 1)
                element_id += 1

    # Generate s values
    edge_elements = identify_edge_elements(elements, N)
    s_values = generate_preload(elements, edge_elements, ratio_outer_to_inner)

    # Define elements preload using s values
    for element_id, s_value in zip(elements.keys(), s_values):
        elements_preload[element_id] = (
            s_value  # Preload force for each element
        )

    return nodes, elements, elements_preload, nodes_load, nodes_fixed


def identify_edge_elements(elements, N):
    """
    Identify the sequence of elements forming the perimeter of the grid.

    Parameters:
    - elements: Dictionary of elements with (start_node, end_node) tuples.
    - N: Grid size (NxN)

    Returns:
    - edge_elements: Ordered list of element IDs forming the boundary.
    """
    # Identify boundary nodes (nodes on the perimeter)
    boundary_nodes = set()
    for node in range(1, N * N + 1):
        i, j = (node - 1) // N, (node - 1) % N
        if i == 0 or j == 0 or i == N - 1 or j == N - 1:
            boundary_nodes.add(node)

    # Identify boundary elements
    edge_elements = []
    for element_id, (n1, n2) in elements.items():
        if n1 in boundary_nodes and n2 in boundary_nodes:
            edge_elements.append(element_id)

    return edge_elements


def generate_preload(elements, edge_elements, ratio_outer_to_inner=1):
    """
    Assign scaled values to elements based on whether they are on the edge.

    Parameters:
    - elements: Dictionary of elements with (start_node, end_node) tuples.
    - edge_elements: List of element IDs forming the boundary (indexed from 1).
    - ratio_outer_to_inner: Scaling factor for edge elements.

    Returns:
    - s_values: Ordered list of scaled values and others are 1.
    """
    s_values = [1] * len(elements)  # Default all elements to 1
    for edge_id in edge_elements:
        s_values[edge_id - 1] = (
            ratio_outer_to_inner  # Adjust edge elements (1-based index)
        )

    return s_values
