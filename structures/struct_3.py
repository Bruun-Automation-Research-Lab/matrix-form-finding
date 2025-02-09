# "An outline of the natural force density method and its
#  extension to quadrilateral elements" Structure


import numpy as np


def generate_struct(N, spacing=2.0):
    nodes = {}
    elements = []
    fixed_nodes = []

    # Define corner heights for interpolation
    corner_z = np.array([[-1.5, 1.5], [1.5, -1.5]])

    # Generate nodes
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

    # Fix only the corner nodes
    corner_ids = [1, N, N * (N - 1) + 1, N * N]  # IDs of the four corners
    fixed_nodes = corner_ids

    # Define external loads (zero by default)
    external_loads = {node_id: (0.0, 0.0, 0.0) for node_id in nodes}

    # Generate elements (grid connectivity)
    elements = [
        ((i * N + j + 1), (i + di) * N + (j + dj) + 1)
        for i in range(N)
        for j in range(N)
        for di, dj in [(1, 0), (0, 1)]
        if i + di < N and j + dj < N
    ]

    return nodes, elements, external_loads, fixed_nodes
