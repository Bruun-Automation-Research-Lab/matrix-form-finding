# "An outline of the natural force density method and its
#  extension to quadrilateral elements" Structure

import numpy as np


def generate_struct(N=5, spacing=2.0):
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

    # Define elements preload (zero by default)
    for element_id in elements:
        elements_preload[element_id] = 0  # Preload force for each element

    return nodes, elements, elements_preload, nodes_load, nodes_fixed
