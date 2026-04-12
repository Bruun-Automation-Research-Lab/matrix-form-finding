import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


def plot_network_views(
    nodes,
    elements,
    nodes_loads,
    nodes_fixed,
    plot_text=False,
    save=False,
    path="./output/network_views.png",
):
    """
    Create a four-view plot: one large 3D view and three small 2D projections.

    Parameters
    ----------
    nodes : ndarray
        Node coordinates, shape (n_nodes, 3).
    elements : ndarray
        Element connectivity, assumed 1-based indexing, shape (n_elements, 2).
    nodes_loads : ndarray
        Nodal loads, shape (n_nodes, ?).
    nodes_fixed : array-like
        Boolean or int mask indicating fixed nodes.
    plot_text : bool, optional
        If True, plot node and element numbers.
    save : bool, optional
        If True, save the figure to `path`.
    path : str, optional
        File path for saving the figure.
    """
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], height_ratios=[3, 1, 1])

    text_offset = 0.07

    # 3D plot
    ax = fig.add_subplot(gs[:, 0], projection="3d")

    # Plot elements with numbers
    for i, (n1, n2) in enumerate(elements - 1):  # convert 1-based to 0-based
        x_vals = [nodes[n1, 0], nodes[n2, 0]]
        y_vals = [nodes[n1, 1], nodes[n2, 1]]
        z_vals = [nodes[n1, 2], nodes[n2, 2]]
        ax.plot(x_vals, y_vals, z_vals, "k-", alpha=0.8)

        if plot_text:
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
            )

    # Plot nodes
    for i, (x, y, z) in enumerate(nodes):
        if nodes_fixed[i]:
            ax.scatter(x, y, z, color="blue", s=20)
        elif np.any(nodes_loads[i] != 0):
            ax.scatter(x, y, z, color="red", s=30)
        else:
            ax.scatter(x, y, z, color="black", s=10)

    if plot_text:
        for i, (x, y, z) in enumerate(nodes):
            if nodes_fixed[i]:
                ax.text(
                    x + text_offset,
                    y + text_offset,
                    z + text_offset,
                    str(i + 1),
                    color="blue",
                    fontsize=7,
                    weight="bold",
                )
            elif np.any(nodes_loads[i] != 0):
                ax.text(
                    x + text_offset,
                    y + text_offset,
                    z + text_offset,
                    str(i + 1),
                    color="red",
                    fontsize=7,
                    weight="bold",
                )
            else:
                ax.text(
                    x + text_offset,
                    y + text_offset,
                    z + text_offset,
                    str(i + 1),
                    color="black",
                    fontsize=7,
                )

    # Axis limits
    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    z_min, z_max = np.min(nodes[:, 2]), np.max(nodes[:, 2])

    ax.set_xlim([min(x_min - 1, -1), max(x_max + 1, 1)])
    ax.set_ylim([min(y_min - 1, -1), max(y_max + 1, 1)])
    ax.set_zlim([min(z_min - 1, -1), max(z_max + 1, 1)])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # XY-plane
    ax_xy = fig.add_subplot(gs[0, 1])
    for n1, n2 in elements:
        x_vals = [nodes[n1 - 1, 0], nodes[n2 - 1, 0]]
        y_vals = [nodes[n1 - 1, 1], nodes[n2 - 1, 1]]
        ax_xy.plot(x_vals, y_vals, "k-")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_title("XY Plane")
    ax_xy.set_aspect("equal")

    # XZ-plane
    ax_xz = fig.add_subplot(gs[1, 1])
    for n1, n2 in elements:
        x_vals = [nodes[n1 - 1, 0], nodes[n2 - 1, 0]]
        z_vals = [nodes[n1 - 1, 2], nodes[n2 - 1, 2]]
        ax_xz.plot(x_vals, z_vals, "k-")
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_title("XZ Plane")
    ax_xz.set_aspect("equal")

    # YZ-plane
    ax_yz = fig.add_subplot(gs[2, 1])
    for n1, n2 in elements:
        y_vals = [nodes[n1 - 1, 1], nodes[n2 - 1, 1]]
        z_vals = [nodes[n1 - 1, 2], nodes[n2 - 1, 2]]
        ax_yz.plot(y_vals, z_vals, "k-")
    ax_yz.set_xlabel("Y")
    ax_yz.set_ylabel("Z")
    ax_yz.set_title("YZ Plane")
    ax_yz.set_aspect("equal")

    plt.tight_layout()

    if save:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_animation(
    node_positions,
    e,
    n_f,
    t=50,
    z_scale=1,
    frame_step=1,
    plot_text=False,
    save=False,
    path="./output/animation.gif",
):
    node_positions = np.asarray(node_positions, dtype=float)
    elements = np.asarray(e, dtype=int) - 1  # convert 1-based to 0-based once
    fixed_nodes = np.asarray(n_f).flatten().astype(bool)

    n_frames, n_nodes, _ = node_positions.shape
    # n_elements = elements.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # --- set global axis limits once ---
    x_all = node_positions[:, :, 0]
    y_all = node_positions[:, :, 1]
    z_all = node_positions[:, :, 2]

    x_min, x_max = np.min(x_all), np.max(x_all)
    y_min, y_max = np.min(y_all), np.max(y_all)
    z_min, z_max = np.min(z_all), np.max(z_all)

    ax.set_xlim([min(x_min, -1), max(x_max, 1)])
    ax.set_ylim([min(y_min, -1), max(y_max, 1)])
    ax.set_zlim([min(z_min, -1), max(z_max, 1)])

    try:
        ax.set_box_aspect(
            [
                max(np.ptp(x_all), 1.0),
                max(np.ptp(y_all), 1.0),
                z_scale * max(np.ptp(z_all), 1.0),
            ]
        )
    except Exception:
        pass

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # --- create artists once ---
    coords0 = node_positions[0]

    node_scatter = ax.scatter(
        coords0[:, 0],
        coords0[:, 1],
        coords0[:, 2],
        c="black",
        marker="o",
        s=10,
        label="Nodes",
    )

    fixed_scatter = ax.scatter(
        coords0[fixed_nodes, 0],
        coords0[fixed_nodes, 1],
        coords0[fixed_nodes, 2],
        c="blue",
        marker="o",
        s=30,
        label="Fixed Nodes",
    )

    element_lines = []
    for start, end in elements:
        (line,) = ax.plot(
            [coords0[start, 0], coords0[end, 0]],
            [coords0[start, 1], coords0[end, 1]],
            [coords0[start, 2], coords0[end, 2]],
            c="green",
            linestyle="-",
            linewidth=1,
        )
        element_lines.append(line)

    node_labels = []
    element_labels = []
    text_offset = 0.05

    if plot_text:
        for i, (x, y, z) in enumerate(coords0):
            txt = ax.text(
                x + text_offset,
                y + text_offset,
                z + text_offset,
                str(i + 1),
                color="blue" if fixed_nodes[i] else "black",
                fontsize=7,
                weight="bold" if fixed_nodes[i] else "normal",
            )
            node_labels.append(txt)

        for i, (start, end) in enumerate(elements):
            mid = 0.5 * (coords0[start] + coords0[end])
            txt = ax.text(
                mid[0] + text_offset,
                mid[1] + text_offset,
                mid[2] + text_offset,
                str(i + 1),
                color="green",
                fontsize=7,
            )
            element_labels.append(txt)

    title = ax.set_title("Iteration 0")
    ax.legend()

    def update(frame):
        coords = node_positions[frame]

        # update node scatter
        node_scatter._offsets3d = (
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
        )

        # update fixed node scatter
        fixed_coords = coords[fixed_nodes]
        fixed_scatter._offsets3d = (
            fixed_coords[:, 0],
            fixed_coords[:, 1],
            fixed_coords[:, 2],
        )

        # update element lines
        for line, (start, end) in zip(element_lines, elements):
            line.set_data(
                [coords[start, 0], coords[end, 0]],
                [coords[start, 1], coords[end, 1]],
            )
            line.set_3d_properties([coords[start, 2], coords[end, 2]])

        # update text
        if plot_text:
            for i, txt in enumerate(node_labels):
                x, y, z = coords[i]
                txt.set_position((x + text_offset, y + text_offset))
                txt.set_3d_properties(z + text_offset, zdir="z")

            for txt, (start, end) in zip(element_labels, elements):
                mid = 0.5 * (coords[start] + coords[end])
                txt.set_position((mid[0] + text_offset, mid[1] + text_offset))
                txt.set_3d_properties(mid[2] + text_offset, zdir="z")

        title.set_text(f"Iteration {frame}")

        artists = [node_scatter, fixed_scatter, title, *element_lines]
        if plot_text:
            artists.extend(node_labels)
            artists.extend(element_labels)
        return artists

    frames = range(0, n_frames, frame_step)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=t,
        blit=False,  # usually keep False for 3D
        repeat=True,
    )

    if save:
        anim.save(path, writer="pillow")

    plt.show()
    return anim


def plot_kinetic_energy(
    KE,
    solver,
    save=False,
    path="./output/kinetic_energy.png",
):

    if solver in ["DR_imp", "DR_leap"]:
        # Create a figure with 1 row and 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        offset = int(np.rint(0.2 * len(KE)))

        # Plot the full time series
        axes[0].plot(KE, label="Kinetic Energy", color="b")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Kinetic Energy")
        axes[0].set_title("Kinetic Energy vs. Iteration (Full Time Series)")
        axes[0].legend()
        axes[0].grid(True)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        # Plot the time series without the first 5 iterations
        axes[1].plot(
            range(offset, len(KE)),
            KE[offset:],
            label="Kinetic Energy",
            color="r",
        )
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Kinetic Energy")
        axes[1].set_title(
            "Kinetic Energy vs. Iteration (After 20% of Iterations)"
        )
        axes[1].legend()
        axes[1].grid(True)
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        # Show the plots side by side
        plt.tight_layout()

        if save:
            plt.savefig(path, dpi=300, bbox_inches="tight")

        plt.show()


def plot_quadratic_interp(
    x,
    y,
    x_interp,
    y_interp,
    q1,
    q2,
    KE_q,
    t=0,
    save=False,
    path="./output/quadratic_interp.png",
):

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
        color="black",
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

    if save:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()
