import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def plot_network3D(nodes, elements, nodes_loads, nodes_fixed):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    text_offset = 0.05  # Adjust this value if necessary

    # Plot elements with numbers
    for i, (n1, n2) in enumerate(
        elements - 1
    ):  # Convert 1-based to 0-based indexing
        x_vals = [nodes[n1, 0], nodes[n2, 0]]
        y_vals = [nodes[n1, 1], nodes[n2, 1]]
        z_vals = [nodes[n1, 2], nodes[n2, 2]]
        ax.plot(x_vals, y_vals, z_vals, "k-", alpha=0.8)

        # Compute midpoint for element numbering
        mid_x = (nodes[n1, 0] + nodes[n2, 0]) / 2
        mid_y = (nodes[n1, 1] + nodes[n2, 1]) / 2
        mid_z = (nodes[n1, 2] + nodes[n2, 2]) / 2
        ax.text(
            mid_x + text_offset,
            mid_y + text_offset,
            mid_z + text_offset,
            str(i + 1),
            color="green",
            fontsize=7,
        )  # Offset element number slightly

    # Plot nodes with numbers
    for i, (x, y, z) in enumerate(nodes):
        if nodes_fixed[i]:  # Fixed node
            ax.scatter(x, y, z, color="blue", s=20)
        elif np.any(nodes_loads[i] != 0):  # Loaded node
            ax.scatter(x, y, z, color="red", s=30)
        else:  # Normal node
            ax.scatter(x, y, z, color="black", s=10)

        # Add offset node number
        ax.text(
            x + text_offset,
            y + text_offset,
            z + text_offset,
            str(i + 1),
            color="black",
            fontsize=7,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def plot_network_animated(ax, nodes, elements, fixed_nodes, iteration):
    ax.cla()  # Clear the axes

    text_offset = 0.05  # Offset for labels to avoid overlap

    # Extract x, y, z coordinates from the nodes array
    x_vals, y_vals, z_vals = nodes[:, 0], nodes[:, 1], nodes[:, 2]

    # Plot nodes
    ax.scatter(
        x_vals, y_vals, z_vals, c="black", marker="o", s=10, label="Nodes"
    )

    # Highlight fixed nodes in blue
    fixed_mask = (
        fixed_nodes.flatten() == 1
    )  # Assuming fixed_nodes is a 1D array with 0/1 values
    ax.scatter(
        x_vals[fixed_mask],
        y_vals[fixed_mask],
        z_vals[fixed_mask],
        c="blue",
        marker="o",
        s=30,
        label="Fixed Nodes",
    )

    # Plot elements with numbering
    for i, (start, end) in enumerate(
        elements - 1
    ):  # Convert 1-based to 0-based indexing
        x_start, y_start, z_start = nodes[start]
        x_end, y_end, z_end = nodes[end]
        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            [z_start, z_end],
            c="green",
            linestyle="-",
            linewidth=1,
        )

        # Compute midpoint for element numbering
        mid_x = (x_start + x_end) / 2
        mid_y = (y_start + y_end) / 2
        mid_z = (z_start + z_end) / 2
        ax.text(
            mid_x + text_offset,
            mid_y + text_offset,
            mid_z + text_offset,
            str(i + 1),
            color="green",
            fontsize=7,
        )  # Element number

    # Plot node numbers
    for i, (x, y, z) in enumerate(nodes):
        ax.text(
            x + text_offset,
            y + text_offset,
            z + text_offset,
            str(i + 1),
            color="black",
            fontsize=7,
        )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"Iteration {iteration}"
    )  # Show iteration number in the title
    ax.legend()


def plot_animation(node_positions, e, n_f):
    # Animation update function
    def update(frame):
        plot_network_animated(ax, node_positions[frame], e, n_f, frame)

    # Create the animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    _ = animation.FuncAnimation(
        fig, update, frames=len(node_positions), interval=30
    )

    plt.show()


def plot_kinetic_energy(KE, solver):

    if solver == "DR":
        plt.figure(figsize=(8, 5))
        plt.plot(KE, label="Kinetic Energy", color="b")
        plt.xlabel("Iteration")
        plt.ylabel("Kinetic Energy")
        plt.title("Kinetic Energy vs. Iteration")
        plt.legend()
        plt.grid(True)
        plt.show()
