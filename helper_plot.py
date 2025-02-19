import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def plot_network3D(nodes, elements, nodes_loads, nodes_fixed):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    text_offset = 0.07  # Adjust this value if necessary

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
            ax.text(
                x + text_offset,
                y + text_offset,
                z + text_offset,
                str(i + 1),
                color="blue",  # Blue text for fixed nodes
                fontsize=7,
                weight="bold",  # Bold text for fixed nodes
            )
        elif np.any(nodes_loads[i] != 0):  # Loaded node
            ax.scatter(x, y, z, color="red", s=30)
            ax.text(
                x + text_offset,
                y + text_offset,
                z + text_offset,
                str(i + 1),
                color="red",  # Red text for loaded nodes
                fontsize=7,
                weight="bold",  # Bold text for loaded nodes
            )
        else:  # Normal node
            ax.scatter(x, y, z, color="black", s=10)
            ax.text(
                x + text_offset,
                y + text_offset,
                z + text_offset,
                str(i + 1),
                color="black",  # Normal black text
                fontsize=7,
            )

    # Calculate axis limits based on the node locations
    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    z_min, z_max = np.min(nodes[:, 2]), np.max(nodes[:, 2])

    # Set axis limits with a minimum of -1 and 1
    ax.set_xlim([min(x_min - 1, -1), max(x_max + 1, 1)])
    ax.set_ylim([min(y_min - 1, -1), max(y_max + 1, 1)])
    ax.set_zlim([min(z_min - 1, -1), max(z_max + 1, 1)])

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
        if fixed_nodes[i] == 1:  # Fixed node
            ax.text(
                x + text_offset,
                y + text_offset,
                z + text_offset,
                str(i + 1),
                color="blue",  # Blue text for fixed nodes
                fontsize=7,
                weight="bold",  # Bold text for fixed nodes
            )
        else:  # Normal node
            ax.text(
                x + text_offset,
                y + text_offset,
                z + text_offset,
                str(i + 1),
                color="black",  # Normal black text
                fontsize=7,
            )

    # Calculate axis limits based on the node locations
    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    z_min, z_max = np.min(nodes[:, 2]), np.max(nodes[:, 2])

    # Set axis limits with a minimum of -1 and 1
    ax.set_xlim([min(x_min, -1), max(x_max, 1)])
    ax.set_ylim([min(y_min, -1), max(y_max, 1)])
    ax.set_zlim([min(z_min, -1), max(z_max, 1)])

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"Iteration {iteration}"
    )  # Show iteration number in the title
    ax.legend()


def plot_animation(node_positions, e, n_f, t):
    # Animation update function
    def update(frame):
        plot_network_animated(ax, node_positions[frame], e, n_f, frame)

    # Create the animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    _ = animation.FuncAnimation(
        fig, update, frames=len(node_positions), interval=t
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


def plot_quadratic_interp(x, y, x_interp, y_interp, q1, q2, KE_q, t=0):

    # Determine max limits considering both series
    x_all = np.concatenate((x, x_interp))
    y_all = np.concatenate((y, y_interp))

    x_margin = 0.1 * (max(x_all) - min(x_all))
    y_margin = 0.2 * (max(y_all) - min(y_all))

    # Plot the data
    plt.plot(x, y, marker="o", linestyle="-", label="Original Data")
    plt.plot(
        x_interp,
        y_interp,
        linestyle="-",
        color="red",
        label="Quadratic Interpolation",
    )
    plt.scatter(q2, KE_q, marker="o", color="red", s=40, label="Max Point")

    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Plot of Energy at Time-Steps")
    plt.title(f"Plot of Energy, Iteration = {t:d}")

    # Set axis limits based on the largest range
    plt.xlim(min(x_all) - x_margin, max(x_all) + x_margin)
    plt.ylim(min(y_all) - y_margin, max(y_all) + y_margin)

    # Set x-axis ticks every 0.1
    plt.xticks(np.arange(min(x_all), max(x_all) + 0.1, 0.1))

    # Highlight specific major ticks
    major_ticks = [0.00, 0.25, 0.5, 0.75, 1.00]
    plt.gca().set_xticks(major_ticks, minor=False)
    plt.gca().tick_params(
        axis="x", which="major", length=8, width=2, color="red"
    )

    # Add bracketed text below the major tick marks
    tick_labels = [
        r"$t - \frac{3\Delta t}{2}$",
        r"$t - \Delta t$",
        r"$t - \frac{\Delta t}{2}$",
        r"$t$",
        r"$t + \frac{\Delta t}{2}$",
    ]

    for tick, label in zip(major_ticks, tick_labels):
        plt.text(
            tick, min(y_all) - y_margin * 0.4, label, ha="center", fontsize=10
        )

    # Add red text for q-values at 0.25, 0.5, 0.75
    q_labels = ["q = 1", "q = 0.5", "q = 0"]
    q_positions = [0.25, 0.5, 0.75]

    for q_pos, q_label in zip(q_positions, q_labels):
        plt.text(
            q_pos,
            min(y_all) - y_margin * 0.8,
            q_label,
            color="red",
            fontsize=8,
            ha="center",
        )

    # Annotate the maximum point
    plt.text(
        q2,
        KE_q + 0.6 * y_margin,
        f"({q2:.3f}, {KE_q:.3e})",
        fontsize=8,
        ha="center",
        color="blue",
    )
    plt.text(
        q2,
        KE_q + 0.2 * y_margin,
        f"({q1:.3f}, {KE_q:.3e})",
        fontsize=8,
        ha="center",
        color="red",
    )

    # Add legend
    plt.legend()

    plt.grid(True)
    plt.show()
