import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from structures.struct_3 import generate_struct
from structures.struct_1 import generate_struct
from helper_matrix import (
    generate_struct_arrays,
    create_connectivity_matrix,
    create_length_matrix,
    create_node_force_vectors,
    create_elastic_stiffness_matrix,
    create_nodal_stiffness_matrix,
    create_force_matrix,
    partition_connectivity_matrix,
    partition_nodes_coordinates,
)
from helper_plot import plot_network3D, plot_network_animated
from helper_log import setup_logging


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

    logging.debug("\nNodes:\n %s", n)
    logging.debug("\nElements:\n %s", e)
    logging.debug("\nElements (Preload):\n %s", e_l)
    logging.debug("\nNodes (Loads):\n %s", n_l)
    logging.debug("\nNodes (Fixed):\n %s", n_f)

    logging.debug("\nConnectivity Matrix:\n %s", connectivity_matrix)
    logging.debug("\nC (free nodes):\n %s", C)
    logging.debug("\nCf (fixed nodes):\n %s", C_f)
    logging.debug("\np_x:\n %s", p_x)
    logging.debug("\np_y:\n %s", p_y)
    logging.debug("\np_z:\n %s", p_z)

    # Set convergence criteria
    TOL = 1e-4
    MAX_ITER = 1000

    FD_factor = 1
    L_0 = np.copy(L)
    F_0 = np.diag(np.copy(e_l).flatten())
    E = np.eye(len(e))
    A = np.eye(len(e))
    h = 0.0001
    v_x = 0
    v_y = 0
    v_z = 0

    # Initialize iteration
    for iteration in range(MAX_ITER):
        logging.debug("\n######################")
        logging.debug("# ITERATION: %s", iteration)
        logging.debug("######################")

        x, y, z, x_f, y_f, z_f = partition_nodes_coordinates(n, n_f)
        l_vec, L = create_length_matrix(n, e)

        if solver == "FD_fixed":
            F = L
            Q = FD_factor * F @ np.linalg.inv(L)

            K = Q

            # Compute matrices
            D = C.T @ K @ C
            D_f = C.T @ K @ C_f

            # Compute new positions
            d_x, d_y, d_z = nodes_delta(
                p_x, p_y, p_z, D, D, D_f, x, y, z, x_f, y_f, z_f
            )

        elif solver == "FD_iter":
            F = FD_factor * np.eye(len(e))
            Q = F @ np.linalg.inv(L)

            K = Q

            # Compute matrices
            D = C.T @ K @ C
            D_f = C.T @ K @ C_f

            # Compute new positions
            d_x, d_y, d_z = nodes_delta(
                p_x, p_y, p_z, D, D, D_f, x, y, z, x_f, y_f, z_f
            )

        elif solver == "DR":
            F = create_force_matrix(L, L_0, E, A, F_0)
            logging.debug("\nFORCE:\n %s", np.diagonal(F))
            Q = F @ np.linalg.inv(L)

            K_g = Q
            K_e = create_elastic_stiffness_matrix(E, A, L_0)
            K = K_g + K_e

            # Compute matrices
            K_mod = C.T @ K @ C
            logging.debug(
                "\nElement Stiffnes (Geometric):\n %s", np.diagonal(K_g)
            )
            logging.debug(
                "\nElement Stiffnes (Elastic):\n %s", np.diagonal(K_e)
            )
            logging.debug("\nStiffnes (free nodes):\n %s", K_mod)

            delta = np.eye(
                K_mod.shape[0]
            )  # Kronecker delta as an identity matrix
            K_mod = K_mod * delta

            # Check that the nodal way is same as the C*K*C way
            # K = compute_nodal_stiffness_force(E, A, L_0, F, L, e, len(n))
            # K_free = K[n_f.flatten() == 0]
            # K_mod = np.diag(K_free)
            # logging.debug("\nStiffnes (MOD2):\n %s", K_mod)

            D = C.T @ Q @ C
            D_f = C.T @ Q @ C_f

            # Compute new positions
            d_x, d_y, d_z = nodes_delta(
                p_x, p_y, p_z, K_mod, D, D_f, x, y, z, x_f, y_f, z_f
            )

            # M = h^2/2 * K, V1 = V0 + h/M * f
            v_x += h * (2 / h**2) * d_x
            v_y += h * (2 / h**2) * d_y
            v_z += h * (2 / h**2) * d_z

            logging.debug("\nv_x:\n %s", v_x)
            logging.debug("\nv_y:\n %s", v_y)
            logging.debug("\nv_z:\n %s", v_z)

            d_x = v_x * h
            d_y = v_y * h
            d_z = v_z * h

        # Update nodes
        n_new = nodes_update(n, d_x, d_y, d_z, n_f)

        # Check for convergence
        l_vec, L_new = create_length_matrix(n_new, e)
        L_total_new = np.sum(l_vec**2)
        max_error = L_total_new - L_total

        node_positions.append(n_new)

        print(
            f"Iteration {iteration + 1}: "
            f"Total Len = {L_total_new:.3f}, "
            f"Max error = {max_error:.3f}"
        )

        logging.debug("\nd_x:\n %s", d_x)
        logging.debug("\nd_y:\n %s", d_y)
        logging.debug("\nd_z:\n %s", d_z)

        logging.debug("\nnodes new:\n %s", n_new)

        logging.debug(f"Total Len = {L_total_new:.3f}, ")
        logging.debug(f"Max error = {max_error:.3f}")

        if np.abs(max_error) < TOL:
            print("Convergence achieved!")
            break

        # Update for next iteration
        n = n_new
        L = L_new
        L_total = L_total_new

        # plot_network3D(n, e, n_l, n_f)

    else:
        print("Max iterations reached without convergence.")

    # Animation update function
    def update(frame):
        plot_network_animated(ax, node_positions[frame], e, n_f, frame)

    # Create the animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    _ = animation.FuncAnimation(
        fig, update, frames=len(node_positions), interval=10
    )

    plt.show()

    f = np.diag(L_new) * np.diag(Q)

    logging.debug("\n######################")
    logging.debug("# COMPLETED")
    logging.debug("######################")

    logging.debug("\nFinal Nodes:\n %s", n_new)
    logging.debug("\nFinal Element Lengths:\n %s", np.diag(L_new))
    logging.debug("\nFinal Element Forces:\n %s", f)
    logging.debug("\nFinal Element Force Densities:\n %s", Q)
    logging.debug("\nFinal Element Forces (f/f_avg):\n %s", f / np.average(f))

    # Plot final network
    plot_network3D(n, e, n_l, n_f)


if __name__ == "__main__":
    main(debug=True, solver="DR")
    # main(debug=True, solver="FD_fixed")
