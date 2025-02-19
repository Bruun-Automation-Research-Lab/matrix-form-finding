import numpy as np
import logging

from structures.struct_2 import generate_struct

from helper_matrix import (
    generate_struct_arrays,
    create_connectivity_matrix,
    create_length_matrix,
    create_node_force_vectors,
    create_elastic_stiffness_matrix,
    compute_kinetic_energy,
    create_nodal_stiffness_matrix,
    create_force_matrix,
    partition_connectivity_matrix,
    partition_nodes_coordinates,
)
from helper_plot import (
    plot_network3D,
    plot_animation,
    plot_kinetic_energy,
    plot_quadratic_interp,
)
from helper_log import (
    setup_logging,
    debug_initial_struct,
    debug_iteration,
    debug_force_and_density,
    debug_stiffness,
    debug_stiffness_FD,
    debug_deltas,
    debug_new_nodes,
    debug_velocity_kinetic_energy,
    debug_energy_peak,
    debug_table,
    debug_table2,
    debug_error,
    debug_final,
)


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


def generate_s(elements, N, ratio_outer_to_inner=1):
    """
    Generate the s array where elements on the boundary have a value based on
    the specified ratio and interior elements have another value.

    Parameters:
    - elements: List of elements (indices of grid elements)
    - N: Size of the NxN grid
    - ratio_outer_to_inner: Ratio of the value on the boundary to the inside

    Returns:
    - s: Array with values based on the ratio for boundary and inside
    """
    # Initialize s with values of 1
    s = np.ones(len(elements))

    # Find the nodes on the boundary
    boundary_nodes = set()
    for i in range(N):
        for j in range(N):
            if i == 0 or j == 0 or i == N - 1 or j == N - 1:
                # Boundary node indices
                boundary_nodes.add(i * N + j)

    # Adjust the values in s based on the boundary and interior
    for idx, element in enumerate(elements):
        if element[0] in boundary_nodes or element[1] in boundary_nodes:
            # Boundary element (set the value based on the ratio)
            s[idx] = ratio_outer_to_inner
        else:
            # Interior element (keep the default value of 1)
            s[idx] = 1

    return s


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


# Main computation
def main(debug=False, solver="FD_fixed"):
    setup_logging(debug)

    # nodes, elements, external_loads, fixed_nodes = generate_struct(5)
    nodes, elements, elements_preload, nodes_load, nodes_fixed = (
        generate_struct()
    )

    n, e, e_l, n_l, n_f = generate_struct_arrays(
        nodes, elements, elements_preload, nodes_load, nodes_fixed
    )

    # Plot network
    plot_network3D(n, e, n_l, n_f)

    # Calculate initial element lengths
    l_vec, L = create_length_matrix(n, e)
    L_total = np.sum(l_vec**2)

    # Generate connectivity matrix
    connectivity_matrix = create_connectivity_matrix(n, e)

    C, C_f = partition_connectivity_matrix(connectivity_matrix, n_f)

    # Compute forces on free nodes
    p_x, p_y, p_z = create_node_force_vectors(n_l, n_f)

    # Create a list to hold node positions at each iteration for animation
    node_positions = []
    node_positions.append(n)  # Store initial position

    # Set convergence criteria
    TOL = 1e-4
    MAX_ITER = 100

    FD_factor = 1
    L_0 = np.copy(L)
    F_0 = np.diag(np.copy(e_l).flatten())
    E = np.eye(len(e))
    A = np.eye(len(e))
    h = 0.1
    v_x = 0
    v_y = 0
    v_z = 0
    gamma = 0.7

    first = True
    KE_prev2 = 0.0  # KE at t-2
    KE_prev = 0.0  # KE at t-1
    KE_history = []

    debug_initial_struct(
        n, e, e_l, n_l, n_f, connectivity_matrix, C, C_f, p_x, p_y, p_z
    )

    # Initialize iteration
    for iteration in range(MAX_ITER):
        debug_iteration(iteration, solver)

        x, y, z, x_f, y_f, z_f = partition_nodes_coordinates(n, n_f)
        l_vec, L = create_length_matrix(n, e)

        if solver == "FD_fixed":
            F = L
            Q = FD_factor * F @ np.linalg.inv(L)
            debug_force_and_density(F, Q)

            # Stiffness matrix
            K = C.T @ Q @ C

            # Force Matrix
            D = C.T @ Q @ C
            D_f = C.T @ Q @ C_f

            debug_stiffness_FD(K, D, D_f)

            # Compute new positions
            d_x, d_y, d_z = nodes_delta(
                p_x, p_y, p_z, K, D, D_f, x, y, z, x_f, y_f, z_f
            )

            debug_deltas(d_x, d_y, d_z)

        elif solver == "FD_iter":
            F = FD_factor * np.eye(len(e))
            Q = F @ np.linalg.inv(L)
            debug_force_and_density(F, Q)

            # Stiffness matrix (free nodes)
            K = C.T @ Q @ C

            # Force Matrix (free and fixed nodes)
            D = C.T @ Q @ C
            D_f = C.T @ Q @ C_f

            debug_stiffness_FD(K, D, D_f)

            # Compute change in x,y,z
            d_x, d_y, d_z = nodes_delta(
                p_x, p_y, p_z, K, D, D_f, x, y, z, x_f, y_f, z_f
            )

            debug_deltas(d_x, d_y, d_z)

        elif solver == "DR":
            F = create_force_matrix(L, L_0, E, A, F_0)
            Q = F @ np.linalg.inv(L)
            debug_force_and_density(F, Q)

            # Element-level stiffness matrices
            K_g = Q
            K_e = create_elastic_stiffness_matrix(E, A, L_0)
            K_total = K_g + K_e

            # Free Nodes stiffness matrices
            K = C.T @ K_total @ C

            # Kronecker delta as an identity matrix
            delta = np.eye(K.shape[0])

            K_mod = K * delta

            debug_stiffness(K_g, K_e, K, K_mod)

            # Check that the nodal, is same as the element (C*K*C) way
            # K = compute_nodal_stiffness_force(E, A, L_0, F, L, e, len(n))
            # K_free = K[n_f.flatten() == 0]
            # K_mod = np.diag(K_free)
            # logging.debug("\nStiffnes (MOD2):\n %s", K_mod)

            D = C.T @ Q @ C
            D_f = C.T @ Q @ C_f

            # Compute new positions
            x, y, z = nodes_delta(
                p_x, p_y, p_z, K_mod, D, D_f, x, y, z, x_f, y_f, z_f
            )

            # M = h^2/2 * K, V1 = V0 + h/M * f (normal)
            # M = h^2/2 * K, V1 = h/2*M * f (first iteration)
            if first:
                # Reset the velocity
                v_x = gamma * h * (1 / h**2) * x
                v_y = gamma * h * (1 / h**2) * y
                v_z = gamma * h * (1 / h**2) * z
                first = True
            else:
                # This is V,t-1 + v,t
                v_x += gamma * h * (2 / h**2) * x
                v_y += gamma * h * (2 / h**2) * y
                v_z += gamma * h * (2 / h**2) * z

            d_x = v_x * h
            d_y = v_y * h
            d_z = v_z * h

            KE = compute_kinetic_energy(K_mod, v_x, v_y, v_z, h)

            debug_velocity_kinetic_energy(v_x, v_y, v_z, KE)
            debug_deltas(d_x, d_y, d_z)
            debug_table2([KE_prev2, KE_prev, KE])

            before = False

            # Check for kinetic energy peak: KE_prev2 < KE_prev > KE
            if KE_prev2 < KE_prev > KE:

                q1 = (KE_prev - KE) / ((KE_prev - KE) - (KE_prev2 - KE_prev))
                q2, KE_q, x_interp, y_interp = quadratic_interp(
                    [KE_prev2, KE_prev, KE]
                )
                plot_quadratic_interp(
                    [0, 0.5, 1.0],
                    [KE_prev2, KE_prev, KE],
                    x_interp,
                    y_interp,
                    q1,
                    q2,
                    KE_q,
                )

                q = q1

                debug_energy_peak(q)

                #     # q,KE_q = quadratic_interpol([KE_prev2, KE_prev, KE])
                #     KE_q = quadratic_interpolate2([KE_prev2, KE_prev, KE],q)
                #     logging.debug("\nq: %s", q)

                d_x -= h * (1 + q) * gamma * v_x + gamma * (q) * x
                d_y -= h * (1 + q) * gamma * v_y + gamma * (q) * y
                d_z -= h * (1 + q) * gamma * v_z + gamma * (q) * z
                debug_deltas(d_x, d_y, d_z)

                # Reset velocities to 0 (kinetic damping)
                first = True

            #     # KE_q = quadratic_interpolate([KE_prev2, KE_prev, KE])

            #     before = debug_table(q, KE_q, [KE_prev2, KE_prev, KE])
            #     # KE = KE_q

            KE_history.append(KE)

            # Shift kinetic energy values for next iteration
            if before:
                KE_prev2 = KE_prev2
                KE_prev = KE
            else:
                KE_prev2 = KE_prev
                KE_prev = KE

        # Update nodes
        n_new = nodes_update(n, d_x, d_y, d_z, n_f)
        node_positions.append(n_new)

        # Check for convergence
        l_vec, L_new = create_length_matrix(n_new, e)
        L_total_new = np.sum(l_vec**2)
        error = np.abs(L_total_new - L_total)

        print(
            f"Iteration {iteration + 1}: "
            f"Total Len = {L_total_new:.3f}, "
            f"Max error = {error:.3e}"
        )

        # Update for next iteration
        n = n_new
        L = L_new
        L_total = L_total_new

        debug_new_nodes(n_new)
        debug_error(L_total, error)

        if error < TOL:
            print("Convergence achieved!")
            break

        # plot_network3D(n, e, n_l, n_f)

    else:
        print("Max iterations reached without convergence.")

    # Final Structure
    debug_final(n_new, L_new, F, Q)

    plot_kinetic_energy(KE_history, solver)

    plot_animation(node_positions, e, n_f)

    plot_network3D(n, e, n_l, n_f)


if __name__ == "__main__":
    # main(debug=True, solver="FD_fixed")
    # main(debug=True, solver="FD_iter")
    main(debug=True, solver="DR")
