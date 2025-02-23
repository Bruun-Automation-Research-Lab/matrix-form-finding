import numpy as np
import logging as log


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


def create_coordinate_diff_matrix(n, C):

    # member coordinate diffs, [m x 3]
    U = C @ n

    return U


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
    C_i = connectivity_matrix[:, free_node_indices]  # Columns for free nodes
    C_f = connectivity_matrix[:, fixed_node_indices]  # Columns for fixed nodes

    return C_i, C_f


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


def create_length_matrix(U):
    """
    Calculate element lengths
    Veenendaal and Block, 2012, implementation
    """

    # member coordinate diffs in x,y,z
    U_bar = np.diag(U[:, 0])
    V_bar = np.diag(U[:, 1])
    W_bar = np.diag(U[:, 2])

    L = np.sqrt(U_bar**2 + V_bar**2 + W_bar**2)

    L_vec = np.diag(L).reshape(-1, 1)

    return L_vec, L


def create_elastic_stiffness_matrix(E, A, L_0):
    """
    Calculate the elastic stiffness matrix K_e for each element.

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

    # Convert to diagonal matrix
    K_e = np.diag(k_e)

    return K_e


def create_stacked_matrices(Q, U, L, C_i):
    """
    Compute the geometric stiffness matrix K_g.

    Parameters:
    Q : ndarray (m x m)
        The force-density diagonal matrix.
    U : ndarray (m x 3)
        The displacement matrix (m rows of [Ux, Uy, Uz]).
    L : ndarray (m x m)
        The length diagonal matrix.
    """

    # Get dimensions of Q
    m, n = Q.shape

    # Create a (3m x 3n) block-diagonal matrix by stacking Q along the diagonal
    Q_stacked = np.zeros((3 * m, 3 * n))
    for i in range(3):
        Q_stacked[i * m : (i + 1) * m, i * n : (i + 1) * n] = Q

    # Reshape U into a single column vector and construct a diagonal matrix
    U_flat = U.reshape(-1, 1)  # Stack x, y, z values in a single column
    U_diag = np.diag(U_flat.flatten())  # Convert to diagonal matrix

    # L to match U’s structure (for x, y, z), construct a diagonal matrix
    L_diag = np.diag(L)  # Convert to diagonal matrix
    L_stacked = np.vstack([L_diag, L_diag, L_diag]).flatten()
    L_diag_expanded = np.diag(L_stacked)

    C_i_stacked = np.vstack([C_i] * 3)

    return Q_stacked, U_diag, L_diag_expanded, C_i_stacked


def create_geometric_stiffnes_SM(Q, U, L):
    """
    Compute the geometric stiffness matrix K_g.

    Parameters:
    Q : ndarray (m x m)
        The force-density diagonal matrix.
    U : ndarray (m x 3)
        The displacement matrix (m rows of [Ux, Uy, Uz]).
    L : ndarray (m x m)
        The length diagonal matrix.

    Returns:
    K_g : ndarray (3m x 3m)
        The geometric stiffness matrix.
    """

    # Compute squared and inverse squared matrices
    U_squared = U @ U  # Equivalent to U^2
    L_squared = L @ L  # Equivalent to L^2
    L_squared_inv = np.linalg.inv(L_squared)  # Equivalent to L^-2

    # Compute geometric stiffness components
    K_base = Q
    K_correction = U_squared @ L_squared_inv @ Q

    # Compute final geometric stiffness matrix
    K_g = K_base - K_correction

    # Debug logs
    log.debug("\nK_g (Base Q_stacked):\n%s", K_base)
    log.debug("\nK_g (Correction Term):\n%s", K_correction)

    return K_g


def create_elastic_stiffness_matrix_SM(E, A, L_0, U, L):
    """
    Calculate the elastic stiffness matrix K_e for each element.

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

    k_e_stacked = np.vstack([k_e, k_e, k_e]).flatten()
    k_e_expanded = np.diag(k_e_stacked)

    # Compute squared and inverse squared matrices
    U_squared = U @ U  # Equivalent to U^2
    L_squared = L @ L  # Equivalent to L^2
    L_squared_inv = np.linalg.inv(L_squared)  # Equivalent to L^-2

    K_e = U_squared @ L_squared_inv @ k_e_expanded

    log.debug("\nE*A/L_0 stacked:\n%s", k_e_expanded)

    return K_e


def create_nodal_stiffness_matrix(E, A, L_0, F, L, elements, num_nodes):
    """
    Create K = Ke + Kg = (EA/L_0) + (F/L) for each node.

    This is not currently used, but used to check C.T x K x C is same

    Parameters (each element):
    E        : np.ndarray (diagonal square matrix) - Young's modulus
    A        : np.ndarray (diagonal square matrix) - X-sectional area
    L_0      : np.ndarray (diagonal square matrix) - Initial length
    F        : np.ndarray (diagonal square matrix) - Force
    L        : np.ndarray (diagonal square matrix) - Current length
    elements : np.ndarray (n x 2) - Element connectivity (1-based indexing)
    num_nodes: int - Total number of nodes

    Returns:
    nodal_values : np.ndarray (num_nodes,)
    """
    # Extract diagonal values
    E_diag = np.diag(E)
    A_diag = np.diag(A)
    L_0_diag = np.diag(L_0)
    F_diag = np.diag(F)
    L_diag = np.diag(L)

    # Compute (EA/L₀) + (F/L) for each element
    element_values = (E_diag * A_diag) / L_0_diag + F_diag / L_diag

    # Initialize nodal contribution array
    nodal_values = np.zeros(num_nodes)

    # Assemble contributions at each node
    for i, (node1, node2) in enumerate(
        elements - 1
    ):  # Convert 1-based to 0-based indexing
        nodal_values[node1] += element_values[i]
        nodal_values[node2] += element_values[i]

    return nodal_values


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
