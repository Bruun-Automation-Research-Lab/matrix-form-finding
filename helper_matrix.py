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


# 1. Create the connectivity matrix
def create_connectivity_matrix_old(nodes, elements):
    num_nodes = len(nodes)
    num_elements = len(elements)

    connectivity_matrix = np.zeros((num_elements, num_nodes), dtype=int)

    for i, (start, end) in enumerate(elements):
        start_idx, end_idx = (
            abs(start) - 1,
            abs(end) - 1,
        )  # Convert to 0-based index
        if start < 0:
            (
                connectivity_matrix[i, end_idx],
                connectivity_matrix[i, start_idx],
            ) = (1, -1)
        else:
            (
                connectivity_matrix[i, start_idx],
                connectivity_matrix[i, end_idx],
            ) = (1, -1)

    return connectivity_matrix


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


def partition_connectivity_matrix_old(connectivity_matrix, nodes, fixed_nodes):
    free_nodes = [node for node in nodes if node not in fixed_nodes]
    all_nodes = list(nodes.keys())

    free_node_indices = [all_nodes.index(node) for node in free_nodes]
    fixed_node_indices = [all_nodes.index(node) for node in fixed_nodes]

    C = connectivity_matrix[:, free_node_indices]
    Cf = connectivity_matrix[:, fixed_node_indices]

    return C, Cf


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


def create_length_matrix_old(nodes, elements):
    lengths = [
        np.linalg.norm(np.array(nodes[abs(start)]) - np.array(nodes[abs(end)]))
        for start, end in elements
    ]
    l_vec = np.array(lengths).reshape(-1, 1)
    L_mat = np.diag(l_vec.flatten())
    return l_vec, L_mat


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


def calculate_force(L, L_0, E, A, F_0):
    EA = E * A  # Compute EA separately
    forces = (EA / L_0) * (L - L_0) + F_0
    F_mat = np.diag(forces.flatten())  # Create diagonal matrix
    return forces, F_mat
