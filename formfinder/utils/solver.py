import numpy as np


def nodes_delta(p_x, p_y, p_z, K, D, D_f, x, y, z, x_f, y_f, z_f):
    # Compute the right-hand side vector (p - D * x - D_f * x_f)
    rhs_x = p_x - D @ x - D_f @ x_f
    rhs_y = p_y - D @ y - D_f @ y_f
    rhs_z = p_z - D @ z - D_f @ z_f

    # Compute the inverse of D
    K_inv = np.linalg.inv(K)

    # Solve for the displacements (delta)
    delta_x = K_inv @ rhs_x
    delta_y = K_inv @ rhs_y
    delta_z = K_inv @ rhs_z

    return delta_x, delta_y, delta_z


def nodes_delta2(p_x, p_y, p_z, K, D, D_f, x, y, z, x_f, y_f, z_f):
    # Compute the right-hand side vector (p - D * x - D_f * x_f)

    p_3x3 = np.vstack([p_x, p_y, p_z])
    x_3x3 = np.vstack([x, y, z])
    x_f_3x3 = np.vstack([x_f, y_f, z_f])

    rhs = p_3x3 - D @ x_3x3 - D_f @ x_f_3x3

    # Solve for the displacements (delta)
    delta_x_3x3 = np.linalg.solve(K, rhs)

    # split back into x, y, z vectors
    n = delta_x_3x3.shape[0] // 3
    delta_x = delta_x_3x3[0:n, :]
    delta_y = delta_x_3x3[n : 2 * n, :]
    delta_z = delta_x_3x3[2 * n : 3 * n, :]

    return delta_x, delta_y, delta_z


def nodes_update(nodes, d_x, d_y, d_z, n_f):
    # Flatten the free_node_mask to match the shape of n_f
    free_node_mask = n_f.flatten() == 0

    # Create a copy of the original nodes array to avoid in-place modification
    updated_nodes = np.copy(nodes)

    # Apply displacement to free nodes
    updated_nodes[free_node_mask, 0] += d_x.flatten()  # Update x-coordinates
    updated_nodes[free_node_mask, 1] += d_y.flatten()  # Update y-coordinates
    updated_nodes[free_node_mask, 2] += d_z.flatten()  # Update z-coordinates

    return updated_nodes


def quadratic_interp(y_points, num_points=100):
    x_points = np.array([0.0, 0.5, 1.0])

    # Fit a quadratic polynomial
    coeffs = np.polyfit(x_points, y_points, 2)

    # Extract the coefficients a, b, and c
    a, b, c = coeffs

    # Find the x-value where the maximum occurs (vertex of the parabola)
    x_max = -b / (2 * a)

    # Find the maximum y-value
    y_max = np.polyval(coeffs, x_max)

    # Generate smooth x values for plotting
    x_interp = np.linspace(min(x_points), max(x_points), num_points)
    y_interp = np.polyval(coeffs, x_interp)

    return x_max, y_max, x_interp, y_interp


def compute_kinetic_energy(K, v_x, v_y, v_z, h):
    """
    Compute the kinetic energy given a diagonal mass matrix K_mod
    and velocity components v_x, v_y, v_z.

    Parameters:
    K_mod : (N, N) numpy.ndarray - Diagonal mass matrix (N x N)
    v_x, v_y, v_z : (N,) numpy.ndarray - Velocity components

    Returns:
    KE : float - Total kinetic energy
    """
    # Extract the diagonal elements (masses)
    masses = np.diag(K) * h**2 / 2

    # Compute kinetic energy for each direction
    KE_x = 0.5 * np.dot(masses, v_x**2)
    KE_y = 0.5 * np.dot(masses, v_y**2)
    KE_z = 0.5 * np.dot(masses, v_z**2)

    # Total kinetic energy
    KE = np.sum(KE_x + KE_y + KE_z)
    return KE
