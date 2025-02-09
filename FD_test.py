import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from generate_grid import generate_grid
from plotting import plot_network3D_2, plot_network3D

from struct_1 import nodes, elements, external_loads, fixed_nodes

# 1. Create the connectivity matrix
def create_connectivity_matrix(nodes, elements):
    num_nodes = len(nodes)
    num_elements = len(elements)

    # Initialize a matrix of size (num_elements x num_nodes)
    connectivity_matrix = np.zeros((num_elements, num_nodes), dtype=int)

    for i, (start, end) in enumerate(elements):
        # Handle negative node indices (reverse direction)
        start_idx = abs(start) - 1  # Convert 1-based index to 0-based
        end_idx = abs(end) - 1  # Convert 1-based index to 0-based

        if start < 0:
            # Reverse direction: start node becomes the ending node
            connectivity_matrix[i, end_idx] = 1
            connectivity_matrix[i, start_idx] = -1
        else:
            # Normal direction: start node becomes the starting node
            connectivity_matrix[i, start_idx] = 1
            connectivity_matrix[i, end_idx] = -1

    return connectivity_matrix


def partition_connectivity_matrix(connectivity_matrix, nodes, fixed_nodes):
    # Get the list of free nodes by excluding fixed nodes
    all_nodes = list(nodes.keys())
    free_nodes = [node for node in all_nodes if node not in fixed_nodes]

    # Mapping of node indices (to match connectivity_matrix)
    free_node_indices = [all_nodes.index(node) for node in free_nodes]
    fixed_node_indices = [all_nodes.index(node) for node in fixed_nodes]

    # Partition the matrix C and Cf
    C = connectivity_matrix[:, free_node_indices]
    Cf = connectivity_matrix[:, fixed_node_indices]

    return C, Cf


import numpy as np


def separate_coordinates(nodes, fixed_nodes):
    # Initialize lists for x, y, z coordinates of free and fixed nodes
    x, y, z = [], [], []
    x_f, y_f, z_f = [], [], []

    # Iterate over all nodes
    for node, (x_coord, y_coord, z_coord) in nodes.items():
        if node in fixed_nodes:
            # Append to fixed nodes lists
            x_f.append(x_coord)
            y_f.append(y_coord)
            z_f.append(z_coord)
        else:
            # Append to free nodes lists
            x.append(x_coord)
            y.append(y_coord)
            z.append(z_coord)

    # Convert lists to numpy arrays with shape (n, 1) for column vectors
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    z = np.array(z).reshape(-1, 1)

    x_f = np.array(x_f).reshape(-1, 1)
    y_f = np.array(y_f).reshape(-1, 1)
    z_f = np.array(z_f).reshape(-1, 1)

    return x, y, z, x_f, y_f, z_f


def calculate_element_lengths(nodes, elements):
    lengths = []

    # Iterate over the elements
    for start, end in elements:
        # Get the coordinates of the start and end nodes
        x1, y1, z1 = nodes[abs(start)]
        x2, y2, z2 = nodes[abs(end)]

        # Calculate the Euclidean distance between the nodes
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        lengths.append(length)

    # Convert lengths list to an nx1 numpy array (column vector)
    l = np.array(lengths).reshape(-1, 1)
    L = np.diag(l.flatten())
    return l, L


def create_and_diagonalize(C, Cf, x, xf, y, yf, z, zf):
    # Step 1: Create vectors u, v, and w
    u = np.dot(C, x) + np.dot(Cf, xf)
    v = np.dot(C, y) + np.dot(Cf, yf)
    w = np.dot(C, z) + np.dot(Cf, zf)

    # Step 2: Diagonalize the vectors
    U = np.diag(u.flatten())
    V = np.diag(v.flatten())
    W = np.diag(w.flatten())

    return U, V, W


def generate_force_densities(L, s):
    """
    Compute force densities q using the formula: q = L^-1 * s

    Parameters:
    L (numpy.ndarray): An n x n matrix.
    s (numpy.ndarray): An n x 1 column vector.

    Returns:
    numpy.ndarray: An n x 1 column vector representing force densities.
    """
    # Ensure L is invertible
    if np.linalg.det(L) == 0:
        raise ValueError("Matrix L is singular and cannot be inverted.")

    # Compute q = L^-1 * s
    q = np.linalg.solve(L, s)  # More stable than np.linalg.inv(L) @ s
    return q


def compute_free_node_forces(nodes, external_loads, fixed_nodes):
    """
    Returns three vectors p_x, p_y, and p_z containing the force components
    of the free nodes (nodes that are not in the fixed_nodes list).

    Parameters:
    nodes (dict): Dictionary mapping node IDs to (x, y, z) coordinates.
    external_loads (dict): Dictionary mapping node IDs to (Fx, Fy, Fz) forces.
    fixed_nodes (list): List of node IDs that are fixed.

    Returns:
    tuple: Three NumPy arrays p_x, p_y, p_z containing the force components
           of the free nodes.
    """
    # Identify free nodes (nodes not in fixed_nodes)
    free_nodes = [node_id for node_id in nodes if node_id not in fixed_nodes]

    # Extract force components for free nodes
    p_x = np.array([external_loads[node][0] for node in free_nodes])
    p_y = np.array([external_loads[node][1] for node in free_nodes])
    p_z = np.array([external_loads[node][2] for node in free_nodes])

    return p_x, p_y, p_z


def compute_new_positions(p_x, p_y, p_z, D, D_f, x_f, y_f, z_f):
    """
    Computes new position vectors x_new, y_new, z_new using the equation:

        x_new = D^-1 * (p_x - D_f * x_f)
        y_new = D^-1 * (p_y - D_f * y_f)
        z_new = D^-1 * (p_z - D_f * z_f)

    Parameters:
    p_x, p_y, p_z (numpy.ndarray): Force components for free nodes.
    D (numpy.ndarray): An (n x n) matrix.
    D_f (numpy.ndarray): An (n x m) matrix.
    x_f, y_f, z_f (numpy.ndarray): Fixed node position vectors.

    Returns:
    tuple: (x_new, y_new, z_new) - Updated position vectors as NumPy arrays.
    """
    # Ensure D is invertible
    if np.linalg.det(D) == 0:
        raise ValueError("Matrix D is singular and cannot be inverted.")

    # Compute the inverse of D
    D_inv = np.linalg.inv(D)

    p_x = p_x.reshape(-1, 1)
    p_y = p_y.reshape(-1, 1)
    p_z = p_z.reshape(-1, 1)

    # Compute new position vectors
    x_new = D_inv @ (p_x - D_f @ x_f)
    y_new = D_inv @ (p_y - D_f @ y_f)
    z_new = D_inv @ (p_z - D_f @ z_f)

    return x_new, y_new, z_new


def update_nodes(nodes, x_new, y_new, z_new, fixed_nodes):
    """
    Updates the positions of free nodes using x_new, y_new, z_new.

    Parameters:
    nodes (dict): Dictionary of nodes {index: (x, y, z)}
    x_new, y_new, z_new (numpy.ndarray): Updated position vectors for free nodes.
    fixed_nodes (list): List of fixed node indices.

    Returns:
    dict: Updated nodes dictionary.
    """
    updated_nodes = nodes.copy()  # Copy original dictionary
    free_nodes = [
        node for node in nodes if node not in fixed_nodes
    ]  # Identify free nodes

    # Update only free nodes with new values
    for i, node in enumerate(free_nodes):
        updated_nodes[node] = (
            x_new[i].item(),
            y_new[i].item(),
            z_new[i].item(),
        )

    return updated_nodes




# # Create and diagonalize
# U, V, W = create_and_diagonalize(C, C_f, x, x_f, y, y_f, z, z_f)

# # Output the results
# print("Diagonal matrix U:\n", U)
# print("Diagonal matrix V:\n", V)
# print("Diagonal matrix W:\n", W)


nodes, elements, external_loads, fixed_nodes = generate_grid(5, spacing=2.5)

print(nodes)

# external_loads[13] = (0.0, 0.0, -1)

plot_network3D_2(nodes, elements, fixed_nodes, external_loads)


s = np.ones(len(elements))
q = np.ones(len(elements))
# q[0] = 5
# q[3] = 10



# Generate connectivity matrix
connectivity_matrix = create_connectivity_matrix(nodes, elements)
print("Connectivity Matrix:\n")
print(connectivity_matrix)


C, C_f = partition_connectivity_matrix(connectivity_matrix, nodes, fixed_nodes)
print("C (Connectivity matrix with free nodes):\n")
print(C)
print("Cf (Connectivity matrix with fixed nodes):\n")
print(C_f)


# Compute force components for free nodes
p_x, p_y, p_z = compute_free_node_forces(nodes, external_loads, fixed_nodes)

# Output results
print("p_x:", p_x)
print("p_y:", p_y)
print("p_z:", p_z)


# Set a convergence tolerance
TOL = 1e-6
MAX_ITER = 100  # Prevent infinite loops

# Compute initial L
length, L = calculate_element_lengths(nodes, elements)
print("Element Lengths:", np.diag(L))

for iteration in range(MAX_ITER):
    # print(f"\nIteration {iteration + 1}")

    x, y, z, x_f, y_f, z_f = separate_coordinates(nodes, fixed_nodes)

    # Compute force densities
    q = generate_force_densities(L, s)
    Q = np.diag(q.flatten())

    # Compute matrices
    D = C.T @ Q @ C
    D_f = C.T @ Q @ C_f

    # Compute new positions
    x_new, y_new, z_new = compute_new_positions(
        p_x, p_y, p_z, D, D_f, x_f, y_f, z_f
    )

    # Update node positions
    updated_nodes = update_nodes(nodes, x_new, y_new, z_new, fixed_nodes)

    # Compute new element lengths
    length_new, L_new = calculate_element_lengths(updated_nodes, elements)

    # Check for convergence
    max_error = np.max(np.abs(L_new - L))
    print(f'Iteration {iteration}: Max error = {max_error}')

    if np.allclose(L, L_new, atol=TOL):
        print("Convergence achieved!")
        break

    # Update L for the next iteration
    nodes = updated_nodes
    L = L_new

    plot_network3D_2(nodes, elements, fixed_nodes, external_loads)

else:
    print("Max iterations reached without convergence.")

# Output final results
print("\nFinal Updated Nodes:")
for node, coords in updated_nodes.items():
    print(f"{node}: {coords}")

print("Final Element Lengths:", np.diag(L_new))

f = np.dot(L_new, q)
print("Final Element Forces:", f)
print("Final Element Forces (normalized):", f/np.average(f))

# Plot the network
plot_network3D_2(updated_nodes, elements, fixed_nodes, external_loads)
