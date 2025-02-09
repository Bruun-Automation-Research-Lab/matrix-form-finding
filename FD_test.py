import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from structures.struct_2 import generate_struct
from plotting import plot_network3D


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
    if np.linalg.det(L) == 0:
        raise ValueError("Matrix L is singular and cannot be inverted.")

    return np.linalg.solve(L, s)


def compute_free_node_forces(nodes, external_loads, fixed_nodes):
    free_nodes = [node for node in nodes if node not in fixed_nodes]

    p_x = np.array([external_loads[node][0] for node in free_nodes])
    p_y = np.array([external_loads[node][1] for node in free_nodes])
    p_z = np.array([external_loads[node][2] for node in free_nodes])

    return p_x, p_y, p_z


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
        updated_nodes[node] = (x_new[i][0], y_new[i][0], z_new[i][0])

    return updated_nodes


def total_len(L):
    return np.dot(np.diag(L).T, np.diag(L))


# Function to plot the network at each step of the iteration
def plot_network_animated(ax, nodes, elements, fixed_nodes):
    ax.cla()  # Clear the axes
    x_vals = [nodes[node][0] for node in nodes]
    y_vals = [nodes[node][1] for node in nodes]
    z_vals = [nodes[node][2] for node in nodes]

    # Plot nodes
    ax.scatter(x_vals, y_vals, z_vals, c="b", marker="o", label="Nodes")

    # Highlight fixed nodes in red
    x_fixed = [nodes[node][0] for node in fixed_nodes]
    y_fixed = [nodes[node][1] for node in fixed_nodes]
    z_fixed = [nodes[node][2] for node in fixed_nodes]
    ax.scatter(
        x_fixed, y_fixed, z_fixed, c="r", marker="x", label="Fixed Nodes"
    )

    # Plot the elements (edges between nodes)
    for start, end in elements:
        x_start, y_start, z_start = nodes[abs(start)]
        x_end, y_end, z_end = nodes[abs(end)]
        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            [z_start, z_end],
            c="g",
            linestyle="-",
            linewidth=1,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()


# Main computation
def main():
    # Generate grid
    nodes, elements, external_loads, fixed_nodes = generate_struct(
        5, spacing=2.5
    )
    print("Nodes:\n", nodes)

    # Calculate initial element lengths
    _, L = calculate_element_lengths(nodes, elements)
    L_total = total_len(L)

    # Plot network
    plot_network3D(nodes, elements, fixed_nodes, external_loads)

    s = np.ones(len(elements))
    q = np.ones(len(elements))

    # Generate connectivity matrix
    connectivity_matrix = create_connectivity_matrix(nodes, elements)
    print("Connectivity Matrix:\n", connectivity_matrix)

    C, C_f = partition_connectivity_matrix(
        connectivity_matrix, nodes, fixed_nodes
    )
    print("C (free nodes):\n", C)
    print("Cf (fixed nodes):\n", C_f)

    # Compute forces on free nodes
    p_x, p_y, p_z = compute_free_node_forces(
        nodes, external_loads, fixed_nodes
    )
    print("p_x:", p_x)
    print("p_y:", p_y)
    print("p_z:", p_z)

    # Set convergence criteria
    TOL = 1e-4
    MAX_ITER = 1000

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Initial plot
    plot_network_animated(ax, nodes, elements, fixed_nodes)

    # Create a list to hold node positions at each iteration for animation
    node_positions = []
    node_positions.append(nodes.copy())  # Store initial position

    # Initialize iteration
    for iteration in range(MAX_ITER):
        x, y, z, x_f, y_f, z_f = separate_coordinates(nodes, fixed_nodes)

        # Generate force densities
        q = generate_force_densities(L, s)
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
            f"\
            Iteration {iteration + 1}:\
            Total Len = {L_total_new},\
            Max error = {max_error}"
        )

        if np.abs(max_error) < TOL:
            print("Convergence achieved!")
            break

        # Update for next iteration
        nodes = updated_nodes
        L = L_new
        L_total = L_total_new

        node_positions.append(nodes.copy())

    else:
        print("Max iterations reached without convergence.")

    # Define the update function for the animation
    def update(frame):
        plot_network_animated(ax, node_positions[frame], elements, fixed_nodes)

    # Create the animation
    _ = animation.FuncAnimation(
        fig, update, frames=len(node_positions), interval=500
    )

    # Display the animation
    plt.show()

    # Final output
    print("\nFinal Updated Nodes:")
    for node, coords in updated_nodes.items():
        print(f"{node}: {coords}")

    print("Final Element Lengths:", np.diag(L_new))

    f = np.dot(L_new, q)
    print("Final Element Forces:", f)
    print("Final Element Forces (normalized):", f / np.average(f))

    # Plot final network
    plot_network3D(updated_nodes, elements, fixed_nodes, external_loads)


if __name__ == "__main__":
    main()
