import numpy as np


def generate_struct_arrays(
    nodes, elements, elements_preload, nodes_load, nodes_fixed
):

    # Convert nodes dictionary to NumPy array
    node_ids = sorted(nodes.keys())
    nodes_array = np.array([nodes[nid] for nid in node_ids])

    # Convert elements dictionary to NumPy array
    element_ids = sorted(elements.keys())
    elements_array = np.array([elements[eid] for eid in element_ids])

    # Convert elements_preload dictionary to NumPy array
    elements_preload_array = np.array(
        [elements_preload[eid] for eid in element_ids]
    ).reshape(-1, 1)

    # Convert nodes_load dictionary to NumPy array
    nodes_load_array = np.array([nodes_load[nid] for nid in node_ids])

    # Convert nodes_fixed dictionary to NumPy array
    nodes_fixed_array = np.array(
        [nodes_fixed[nid] for nid in node_ids]
    ).reshape(-1, 1)

    return (
        nodes_array,
        elements_array,
        elements_preload_array,
        nodes_load_array,
        nodes_fixed_array,
    )


def create_connectivity_matrix(nodes, elements):
    num_nodes = nodes.shape[0]
    num_elements = elements.shape[0]

    connectivity_matrix = np.zeros((num_elements, num_nodes), dtype=int)

    # Convert elements from 1-based to 0-based indexing
    elements = np.abs(elements) - 1

    # Assign 1 and -1 values to the correct locations
    for i, (start_idx, end_idx) in enumerate(elements):
        connectivity_matrix[i, start_idx] = 1
        connectivity_matrix[i, end_idx] = -1

    return connectivity_matrix


def partition_connectivity_matrix(connectivity_matrix, fixed_nodes):
    # Flatten the fixed_nodes array to 1D
    fixed_nodes_mask = fixed_nodes.flatten()

    # Get indices of free and fixed nodes
    free_node_indices = np.where(fixed_nodes_mask == 0)[0]
    fixed_node_indices = np.where(fixed_nodes_mask == 1)[0]

    # Partition the connectivity matrix into free and fixed node columns
    C = connectivity_matrix[:, free_node_indices]  # Columns for free nodes
    Cf = connectivity_matrix[:, fixed_node_indices]  # Columns for fixed nodes

    return C, Cf


def partition_nodes_coordinates(nodes, fixed_nodes):
    # Get the indices of fixed and free nodes based on fixed_nodes mask
    fixed_nodes_mask = fixed_nodes.flatten()

    # Indices of free and fixed nodes
    free_node_indices = np.where(fixed_nodes_mask == 0)[0]
    fixed_node_indices = np.where(fixed_nodes_mask == 1)[0]

    # Separate coordinates for free and fixed nodes
    free_coords = nodes[free_node_indices]
    fixed_coords = nodes[fixed_node_indices]

    # Separate into x, y, z components for both free and fixed nodes
    free_x, free_y, free_z = (
        free_coords[:, 0],
        free_coords[:, 1],
        free_coords[:, 2],
    )
    fixed_x, fixed_y, fixed_z = (
        fixed_coords[:, 0],
        fixed_coords[:, 1],
        fixed_coords[:, 2],
    )

    return (
        free_x.reshape(-1, 1),
        free_y.reshape(-1, 1),
        free_z.reshape(-1, 1),
        fixed_x.reshape(-1, 1),
        fixed_y.reshape(-1, 1),
        fixed_z.reshape(-1, 1),
    )


def create_node_force_vectors(nodes_load, nodes_fixed):
    # Flatten fixed_nodes to a 1D array to easily index free/fixed nodes
    fixed_nodes_mask = nodes_fixed.flatten()

    # Get the indices of free nodes (those with fixed_nodes_mask == 0)
    free_node_indices = np.where(fixed_nodes_mask == 0)[0]

    # Extract the loads for free nodes
    p_x = nodes_load[free_node_indices, 0]
    p_y = nodes_load[free_node_indices, 1]
    p_z = nodes_load[free_node_indices, 2]

    return p_x.reshape(-1, 1), p_y.reshape(-1, 1), p_z.reshape(-1, 1)


def create_length_matrix(nodes, elements):
    # Adjust indices (convert from 1-based to 0-based)
    elements = elements - 1

    # Compute lengths
    lengths = np.linalg.norm(
        nodes[elements[:, 0]] - nodes[elements[:, 1]], axis=1
    )

    # Convert to column vector
    l_vec = lengths.reshape(-1, 1)

    # Create diagonal matrix
    L_mat = np.diag(l_vec.flatten())

    return l_vec, L_mat


def create_elastic_stiffness_matrix(E, A, L_0, elements, num_nodes):
    """
    Calculate the global stiffness matrix (diagonal) K_g.

    Parameters:
    E        : np.ndarray (diagonal matrix) - Young's modulus matrix
    A        : np.ndarray (diagonal matrix) - Cross-sectional area matrix
    L_0      : np.ndarray (n,) - Initial length of each element
    elements : np.ndarray (n x 2) - Element connectivity matrix
    nodes    : np.ndarray (num_nodes x 2) - Node coordinate matrix
    num_nodes: int - Total number of nodes in the system

    Returns:
    K_g : np.ndarray (diagonal matrix) - Global stiffness matrix
    """
    # Extract diagonal values from matrices E, A
    E_diag = np.diag(E)
    A_diag = np.diag(A)
    L_0_diag = np.diag(L_0)

    # Compute element stiffness values (E*A / L_0 for each element)
    k_e = (E_diag * A_diag) / L_0_diag

    # # Initialize nodal stiffness vector
    # K_g_vec = np.zeros(num_nodes)

    # # Assemble global stiffness by summing contributions at each node
    # for i, (node1, node2) in enumerate(elements):
    #     # Convert 1-based indexing to 0-based indexing
    #     node1 -= 1
    #     node2 -= 1

    #     K_g_vec[node1] += k_e[i]
    #     K_g_vec[node2] += k_e[i]

    # Convert to diagonal matrix
    K_g = np.diag(k_e)

    return K_g


def create_force_matrix(L, L_0, E, A, F_0):
    """
    Calculate the force matrix F from given diagonal matrices.

    Parameters:
    L    : np.ndarray (diagonal square matrix) - Current length matrix
    L_0  : np.ndarray (diagonal square matrix) - Initial length matrix
    E    : float - Young's modulus
    A    : float - Cross-sectional area
    F_0  : np.ndarray (diagonal square matrix) - Initial force matrix

    Returns:
    F : np.ndarray (diagonal square matrix) - Resulting force matrix
    """
    # Ensure inputs are diagonal matrices
    if not (
        np.allclose(L, np.diag(np.diag(L)))
        and np.allclose(L_0, np.diag(np.diag(L_0)))
        and np.allclose(F_0, np.diag(np.diag(F_0)))
    ):
        raise ValueError("All input matrices must be square and diagonal.")

    # Extract diagonal elements as vectors
    L_diag = np.diag(L)
    E_diag = np.diag(E)
    A_diag = np.diag(A)
    L_0_diag = np.diag(L_0)
    F_0_diag = np.diag(F_0)

    # Compute force vector
    forces = (E_diag * A_diag / L_0_diag) * (L_diag - L_0_diag) + F_0_diag

    # # Return as diagonal matrix
    # F = np.diag(forces)

    return np.diag(forces)
