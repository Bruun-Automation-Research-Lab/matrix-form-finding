# "A Framework for Comparing Form Finding Methods" Structure

import numpy as np


def generate_struct(N, spacing=2.5):
    nodes = {}
    elements = []
    fixed_nodes = []

    # Define corner heights for interpolation
    corner_z = np.array([[-4, 4], [4, -4]])

    # Generate nodes
    node_id = 1
    for i in range(N):
        for j in range(N):
            x, y = i * spacing, j * spacing

            # Compute z-values based on edge/corner location
            if i in {0, N - 1} and j in {0, N - 1}:  # Corners
                z = float(corner_z[i // (N - 1), j // (N - 1)])
            elif i in {0, N - 1}:  # Edge along x-direction
                z = float(np.interp(j, [0, N - 1], corner_z[i // (N - 1), :]))
            elif j in {0, N - 1}:  # Edge along y-direction
                z = float(np.interp(i, [0, N - 1], corner_z[:, j // (N - 1)]))
            else:
                z = 0.0  # Interior nodes

            nodes[node_id] = (x, y, z)

            # Fix outer nodes
            if i == 0 or j == 0 or i == N - 1 or j == N - 1:
                fixed_nodes.append(node_id)

            node_id += 1

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
