import numpy as np


def generate_grid(N, spacing=2.0):
    nodes = {}
    elements = []
    external_loads = {}
    fixed_nodes = []

    node_id = 1
    corner_z = [[-4, 4], [4, -4]]  # Alternating corner heights

    for i in range(N):
        for j in range(N):
            x = i * spacing
            y = j * spacing

            # Set Z values
            if (i == 0 or i == N - 1) and (j == 0 or j == N - 1):
                z = corner_z[i // (N - 1)][
                    j // (N - 1)
                ]  # Assign corner heights
            elif i == 0 or i == N - 1 or j == 0 or j == N - 1:
                # Linearly interpolate Z between corners
                if i == 0 or i == N - 1:
                    z = (
                        (N - 1 - j) * corner_z[i // (N - 1)][0]
                        + j * corner_z[i // (N - 1)][1]
                    ) / (N - 1)
                else:
                    z = (
                        (N - 1 - i) * corner_z[0][j // (N - 1)]
                        + i * corner_z[1][j // (N - 1)]
                    ) / (N - 1)
            else:
                z = 0.0  # Interior nodes at z=0

            nodes[node_id] = (x, y, z)

            # Define external loads (set to zero)
            external_loads[node_id] = (0.0, 0.0, 0.0)

            # Fix outer edge nodes
            if i == 0 or j == 0 or i == N - 1 or j == N - 1:
                fixed_nodes.append(node_id)

            node_id += 1

    def node_index(i, j):
        return i * N + j + 1

    for i in range(N):
        for j in range(N):
            if i < N - 1:
                elements.append((node_index(i, j), node_index(i + 1, j)))
            if j < N - 1:
                elements.append((node_index(i, j), node_index(i, j + 1)))

    return nodes, elements, external_loads, fixed_nodes


# # Example usage
# N = 3  # Define grid size
# nodes, elements, external_loads, fixed_nodes = generate_grid(N)

# print("Nodes:")
# print(nodes)
# print("\nElements:")
# print(elements)
# print("\nExternal Loads:")
# print(external_loads)
# print("\nFixed Nodes:")
# print(fixed_nodes)
