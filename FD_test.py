import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from structures.struct_2 import generate_struct
from plotting import plot_network3D, plot_network_animated


# Set up logging
def setup_logging(debug=False):
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if debug:
        logging.basicConfig(
            filename="./debug_log.txt",
            level=logging.DEBUG,
            format="%(message)s",
            filemode="w",
        )
    else:
        logging.basicConfig(level=logging.INFO)


# 1. Create the connectivity matrix
def create_connectivity_matrix(nodes, elements):
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


def partition_connectivity_matrix(connectivity_matrix, nodes, fixed_nodes):
    free_nodes = [node for node in nodes if node not in fixed_nodes]
    all_nodes = list(nodes.keys())

    free_node_indices = [all_nodes.index(node) for node in free_nodes]
    fixed_node_indices = [all_nodes.index(node) for node in fixed_nodes]

    C = connectivity_matrix[:, free_node_indices]
    Cf = connectivity_matrix[:, fixed_node_indices]

    return C, Cf


def separate_coordinates(nodes, fixed_nodes):
    free_coords, fixed_coords = [], []

    for node, (x, y, z) in nodes.items():
        coords = (x, y, z)
        (fixed_coords if node in fixed_nodes else free_coords).append(coords)

    # Separate into x, y, z for both free and fixed nodes
    free_x, free_y, free_z = zip(*free_coords) if free_coords else ([], [], [])
    fixed_x, fixed_y, fixed_z = (
        zip(*fixed_coords) if fixed_coords else ([], [], [])
    )

    return (
        np.array(free_x).reshape(-1, 1),
        np.array(free_y).reshape(-1, 1),
        np.array(free_z).reshape(-1, 1),
        np.array(fixed_x).reshape(-1, 1),
        np.array(fixed_y).reshape(-1, 1),
        np.array(fixed_z).reshape(-1, 1),
    )


def calculate_element_lengths(nodes, elements):
    lengths = [
        np.linalg.norm(np.array(nodes[abs(start)]) - np.array(nodes[abs(end)]))
        for start, end in elements
    ]
    l_vec = np.array(lengths).reshape(-1, 1)
    L_mat = np.diag(l_vec.flatten())
    return l_vec, L_mat


def generate_force_densities(L, s):
    # if np.linalg.det(L) == 0:
    #     raise ValueError("Matrix L is singular and cannot be inverted.")

    return np.linalg.solve(L, s)


def compute_free_node_forces(nodes, external_loads, fixed_nodes):
    free_nodes = [node for node in nodes if node not in fixed_nodes]

    p_x = np.array([external_loads[node][0] for node in free_nodes])
    p_y = np.array([external_loads[node][1] for node in free_nodes])
    p_z = np.array([external_loads[node][2] for node in free_nodes])

    return p_x.reshape(-1, 1), p_y.reshape(-1, 1), p_z.reshape(-1, 1)


def compute_new_positions(p_x, p_y, p_z, D, D_f, x_f, y_f, z_f):
    if np.linalg.det(D) == 0:
        raise ValueError("Matrix D is singular and cannot be inverted.")

    D_inv = np.linalg.inv(D)
    return (
        D_inv @ (p_x - D_f @ x_f),
        D_inv @ (p_y - D_f @ y_f),
        D_inv @ (p_z - D_f @ z_f),
    )


def update_nodes(nodes, x_new, y_new, z_new, fixed_nodes):
    updated_nodes = nodes.copy()
    free_nodes = [node for node in nodes if node not in fixed_nodes]

    for i, node in enumerate(free_nodes):
        updated_nodes[node] = (
            float(x_new[i]),
            float(y_new[i]),
            float(z_new[i]),
        )

    return updated_nodes


def total_len(L):
    return np.dot(np.diag(L).T, np.diag(L))


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
def main(debug=False):
    setup_logging(debug)
    # Generate grid
    # nodes, elements, external_loads, fixed_nodes = generate_struct(
    #     20, spacing=0.5
    # )

    nodes, elements, external_loads, fixed_nodes = generate_struct(5)

    # Calculate initial element lengths
    _, L = calculate_element_lengths(nodes, elements)
    L_total = total_len(L)

    # Plot network
    plot_network3D(nodes, elements, fixed_nodes, external_loads)

    # s = np.ones(len(elements))
    # s = generate_s(elements, 20, 5)

    q = np.ones(len(elements))
    q = generate_s(elements, 20, 1)

    # Generate connectivity matrix
    connectivity_matrix = create_connectivity_matrix(nodes, elements)

    C, C_f = partition_connectivity_matrix(
        connectivity_matrix, nodes, fixed_nodes
    )

    # Compute forces on free nodes
    p_x, p_y, p_z = compute_free_node_forces(
        nodes, external_loads, fixed_nodes
    )

    # Set convergence criteria
    TOL = 1e-4
    MAX_ITER = 1000

    # Create a list to hold node positions at each iteration for animation
    node_positions = []
    node_positions.append(nodes.copy())  # Store initial position

    logging.debug("Nodes:\n %s", nodes)
    logging.debug("\nConnectivity Matrix:\n %s", connectivity_matrix)
    logging.debug("\nC (free nodes):\n %s", C)
    logging.debug("\nCf (fixed nodes):\n %s", C_f)
    logging.debug("\np_x:\n %s", p_x)
    logging.debug("\np_y:\n %s", p_y)
    logging.debug("\np_z:\n %s", p_z)

    # Initialize iteration
    for iteration in range(MAX_ITER):
        x, y, z, x_f, y_f, z_f = separate_coordinates(nodes, fixed_nodes)

        # Generate force densities
        # q = generate_force_densities(L, s)
        Q = np.diag(q.flatten())

        # Compute matrices
        D = C.T @ Q @ C
        D_f = C.T @ Q @ C_f

        # Compute new positions
        x_new, y_new, z_new = compute_new_positions(
            p_x, p_y, p_z, D, D_f, x_f, y_f, z_f
        )

        # Update nodes
        updated_nodes = update_nodes(nodes, x_new, y_new, z_new, fixed_nodes)

        # Check for convergence
        _, L_new = calculate_element_lengths(updated_nodes, elements)
        L_total_new = total_len(L_new)
        max_error = L_total_new - L_total

        print(
            f"Iteration {iteration + 1}: "
            f"Total Len = {L_total_new:.3f}, "
            f"Max error = {max_error:.3f}"
        )

        if np.abs(max_error) < TOL:
            print("Convergence achieved!")
            break

        # Update for next iteration
        nodes = updated_nodes
        L = L_new
        L_total = L_total_new

        node_positions.append(nodes.copy())

        plot_network3D(nodes, elements, fixed_nodes, external_loads)

    else:
        print("Max iterations reached without convergence.")

    # Animation
    def update(frame):
        plot_network_animated(ax, node_positions[frame], elements, fixed_nodes)

    # Create the animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    _ = animation.FuncAnimation(
        fig, update, frames=len(node_positions), interval=500
    )

    plt.show()

    f = np.dot(L_new, q)

    logging.debug("\nFinal Nodes:\n %s", updated_nodes)
    logging.debug("\nFinal Element Lengths:\n %s", np.diag(L_new))
    logging.debug("\nFinal Element Forces:\n %s", f)
    logging.debug("\nFinal Element Force Densities:\n %s", q)
    logging.debug("\nFinal Element Forces (f/f_avg):\n %s", f / np.average(f))

    # Plot final network
    plot_network3D(updated_nodes, elements, fixed_nodes, external_loads)


if __name__ == "__main__":
    main(debug=True)
