import numpy as np

import helper_log as hl
import helper_solver as hs
import helper_matrix as hm
import helper_plot as hp

from structures.struct_4 import generate_struct


# Main computation
def main(debug=False, solver="FD_fixed"):
    hl.setup_logging(debug)

    # nodes, elements, external_loads, fixed_nodes = generate_struct(5)
    nodes, elements, elements_preload, nodes_load, nodes_fixed = (
        generate_struct()
    )

    n, e, e_l, n_l, n_f = hm.generate_struct_arrays(
        nodes, elements, elements_preload, nodes_load, nodes_fixed
    )

    # Plot network
    hp.plot_network_views(n, e, n_l, n_f)

    # Calculate initial element lengths
    l_vec, L = hm.create_length_matrix(n, e)
    L_total = np.sum(l_vec**2)

    # Generate connectivity matrix
    connectivity_matrix = hm.create_connectivity_matrix(n, e)

    C, C_f = hm.partition_connectivity_matrix(connectivity_matrix, n_f)

    # Compute forces on free nodes
    p_x, p_y, p_z = hm.create_node_force_vectors(n_l, n_f)

    # Create a list to hold node positions at each iteration for animation
    node_positions = []
    node_positions.append(n)  # Store initial position

    # Set convergence criteria
    TOL = 1e-4
    MAX_ITER = 10000

    FD_factor = 1
    L_0 = np.copy(L)
    F_0 = np.diag(np.copy(e_l).flatten())
    E = np.eye(len(e))
    A = np.eye(len(e))

    # DR variables
    h = 0.5
    v_x = 0
    v_y = 0
    v_z = 0
    gamma = 1.0

    first = True
    KE_prev2 = 0.0  # KE at t-2
    KE_prev = 0.0  # KE at t-1
    KE_history = []

    hl.debug_initial_struct(
        n, e, e_l, n_l, n_f, connectivity_matrix, C, C_f, p_x, p_y, p_z
    )

    # Initialize iteration
    for iteration in range(MAX_ITER):
        hl.debug_iteration(iteration, solver)

        x, y, z, x_f, y_f, z_f = hm.partition_nodes_coordinates(n, n_f)
        l_vec, L = hm.create_length_matrix(n, e)

        if solver == "FD_fixed":
            F = L
            Q = FD_factor * F @ np.linalg.inv(L)
            # Q = np.diag(Q @ e_l.flatten()) # A way to apply pre-load
            hl.debug_force_and_density(F, Q)

            # Stiffness matrix
            K = C.T @ Q @ C

            # Force Matrix
            D = C.T @ Q @ C
            D_f = C.T @ Q @ C_f

            hl.debug_stiffness_FD(K, D, D_f)

            # Compute new positions
            d_x, d_y, d_z = hs.nodes_delta(
                p_x, p_y, p_z, K, D, D_f, x, y, z, x_f, y_f, z_f
            )

            hl.debug_deltas(d_x, d_y, d_z)

        elif solver == "FD_iter":
            F = FD_factor * np.eye(len(e))
            Q = F @ np.linalg.inv(L)
            hl.debug_force_and_density(F, Q)

            # Stiffness matrix (free nodes)
            K = C.T @ Q @ C

            # Force Matrix (free and fixed nodes)
            D = C.T @ Q @ C
            D_f = C.T @ Q @ C_f

            hl.debug_stiffness_FD(K, D, D_f)

            # Compute change in x,y,z
            d_x, d_y, d_z = hs.nodes_delta(
                p_x, p_y, p_z, K, D, D_f, x, y, z, x_f, y_f, z_f
            )

            hl.debug_deltas(d_x, d_y, d_z)

        elif solver == "DR":
            F = hm.create_force_matrix(L, L_0, E, A, F_0)
            Q = F @ np.linalg.inv(L)
            hl.debug_force_and_density(F, Q)

            # Element-level stiffness matrices
            K_g = Q
            K_e = hm.create_elastic_stiffness_matrix(E, A, L_0)
            K_total = K_g + K_e

            # Free Nodes stiffness matrices
            # This is same as if you summed each element at each node
            K = C.T @ K_total @ C

            # Kronecker delta as an identity matrix
            # This seems to destabilize when doing leapfrog integration
            # delta = np.eye(K.shape[0])
            # K_mod = K * delta
            K_mod = K

            hl.debug_stiffness(K_g, K_e, K, K_mod)

            D = C.T @ Q @ C
            D_f = C.T @ Q @ C_f

            # Compute new positions
            x, y, z = hs.nodes_delta(
                p_x, p_y, p_z, K_mod, D, D_f, x, y, z, x_f, y_f, z_f
            )

            # M = h^2/2 * K, V1 = V0 + h/M * f (normal)
            # M = h^2/2 * K, V1 = h/2*M * f (first iteration)
            if first:
                # Reset the velocity
                v_x = gamma * h * (1 / h**2) * x
                v_y = gamma * h * (1 / h**2) * y
                v_z = gamma * h * (1 / h**2) * z
                first = False
            else:
                # This is V,t-1 + v,t
                v_x += gamma * h * (2 / h**2) * x
                v_y += gamma * h * (2 / h**2) * y
                v_z += gamma * h * (2 / h**2) * z

            d_x = v_x * h
            d_y = v_y * h
            d_z = v_z * h

            KE = hs.compute_kinetic_energy(K_mod, v_x, v_y, v_z, h)

            hl.debug_velocity_kinetic_energy(v_x, v_y, v_z, KE)
            hl.debug_deltas(d_x, d_y, d_z)
            hl.debug_table([KE_prev2, KE_prev, KE])

            check_KE = True

            # Check for kinetic energy peak: KE_prev2 < KE_prev > KE
            if KE_prev2 < KE_prev > KE and check_KE:

                q1 = (KE_prev - KE) / ((KE_prev - KE) - (KE_prev2 - KE_prev))

                # # this is same interp, different interval than the paper
                # q2, KE_q, x_interp, y_interp = hs.quadratic_interp(
                #     [KE_prev2, KE_prev, KE]
                # )
                # hp.plot_quadratic_interp(
                #     [0, 0.5, 1.0],
                #     [KE_prev2, KE_prev, KE],
                #     x_interp,
                #     y_interp,
                #     q1,
                #     q2,
                #     KE_q,
                #     t=iteration,
                # )

                q = q1
                hl.debug_energy_peak(q1)

                d_x -= h * (1 + q) * gamma * v_x + gamma * (q) * x
                d_y -= h * (1 + q) * gamma * v_y + gamma * (q) * y
                d_z -= h * (1 + q) * gamma * v_z + gamma * (q) * z
                hl.debug_deltas(d_x, d_y, d_z)

                # Reset velocities to 0 (kinetic damping)
                first = True

                # _ = hl.debug_table(q1, KE_q, [KE_prev2, KE_prev, KE])
                # KE = np.copy(KE_q)

            KE_prev2 = np.copy(KE_prev)
            KE_prev = np.copy(KE)

            KE_history.append(KE)

        # Update nodes
        n_new = hs.nodes_update(n, d_x, d_y, d_z, n_f)
        node_positions.append(n_new)

        # Check for convergence
        l_vec, L_new = hm.create_length_matrix(n_new, e)
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

        hl.debug_new_nodes(n_new)
        hl.debug_error(L_total, error)

        if error < TOL:
            print("Convergence achieved!")
            break

        # hp.plot_network3D(n, e, n_l, n_f)

    else:
        print("Max iterations reached without convergence.")

    # Final Structure
    hl.debug_final(n_new, L_new, F, Q)

    hp.plot_kinetic_energy(KE_history, solver)

    hp.plot_animation(node_positions, e, n_f, t=1)

    hp.plot_network_views(n, e, n_l, n_f)


if __name__ == "__main__":
    # main(debug=True, solver="FD_fixed")
    # main(debug=True, solver="FD_iter")
    main(debug=True, solver="DR")

# DR with no preload is basically FD_fixed with Q = 1
