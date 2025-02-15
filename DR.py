import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from structures.struct_1 import generate_struct
from helper_plot import plot_network3D


class DynamicRelaxation:
    def __init__(
        self,
        nodes,
        elements,
        fixed,
        mass=1.0,
        damping=0.90,
        dt=0.1,
        tol=1e-3,
        external_forces=None,
    ):
        # Convert the nodes dictionary into a NumPy array
        self.nodes = np.array(
            list(nodes.values()), dtype=float
        )  # Node coordinates
        self.elements = np.array(elements, dtype=int)  # connectivity
        self.fixed = fixed  # Fixed nodes (list of node indices)
        self.mass = mass  # Mass per node
        self.damping = damping  # Damping factor
        self.dt = dt  # Time step
        self.tol = tol  # Convergence tolerance
        self.velocities = np.zeros_like(self.nodes)  # Initial velocities
        # Convert external forces dictionary into a NumPy array
        self.external_forces = (
            np.array(
                [external_forces.get(i, (0.0, 0.0, 0.0)) for i in nodes.keys()]
            )
            if external_forces is not None
            else np.zeros_like(self.nodes)
        )
        self.frames = []  # Store frames for animation

    def length(self, edge):
        """Compute length of an edge."""
        return np.linalg.norm(
            self.nodes[edge[1] - 1] - self.nodes[edge[0] - 1]
        )

    def solve(self, target_length=None, save_every_n=20):
        """Iteratively solve for equilibrium."""
        converged = False
        iteration = 0
        forces = np.zeros_like(self.nodes)
        target_length = target_length or {
            tuple(e): self.length(e) for e in self.elements
        }

        self.frames.append(np.copy(self.nodes))

        while not converged:
            forces.fill(0)  # Reset forces

            # Compute internal forces
            for edge in self.elements:
                i, j = edge
                vec = self.nodes[j - 1] - self.nodes[i - 1]
                length = np.linalg.norm(vec)
                direction = vec / (length + 1e-9)
                force_magnitude = length - target_length[tuple(edge)]
                forces[i - 1] += force_magnitude * direction
                forces[j - 1] -= force_magnitude * direction

            # Apply external forces
            forces += self.external_forces

            # Apply boundary conditions (fix the velocity of fixed nodes)
            forces[np.array(self.fixed) - 1] = 0

            # Update velocities and positions
            self.velocities += (forces / self.mass) * self.dt
            self.velocities *= self.damping
            self.nodes += self.velocities * self.dt

            # Store frames for animation
            if iteration % save_every_n == 0:
                self.frames.append(np.copy(self.nodes))

            # Check for convergence
            max_displacement = np.max(np.abs(self.velocities))

            # Print iteration info every 100 steps
            if iteration % 100 == 0:
                print(
                    f"Iteration {iteration:6d} | "
                    f"Max Displacement: {max_displacement:.3e}"
                )

            if max_displacement < self.tol:
                converged = True

            iteration += 1
            if iteration > 10000:
                print("Warning: Max iterations reached")
                break

        print(f"Converged in {iteration} iterations.")
        return self.nodes

    def plot_structure(self, nodes, original_nodes=None):

        edges = self.elements
        fixed = self.fixed
        external_forces = self.external_forces

        """Plot the structural configuration in 3D."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Plot original structure if provided
        if original_nodes is not None:
            for edge in edges:
                ax.plot(
                    *zip(*np.array(original_nodes)[edge - 1]),
                    color="gray",
                    linestyle="dashed",
                    linewidth=0.5,
                    alpha=0.5,
                )

        # Plot final structure
        for edge in edges:
            ax.plot(*zip(*nodes[edge - 1]), color="black")

        # Identify nodes with applied loads
        loaded_nodes = np.any(external_forces != 0, axis=1)

        # Color nodes based on type
        nodes = np.array(nodes)
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            nodes[:, 2],
            color="black",
            label="Free Nodes",
        )
        ax.scatter(
            nodes[np.array(fixed) - 1, 0],
            nodes[np.array(fixed) - 1, 1],
            nodes[np.array(fixed) - 1, 2],
            color="red",
            label="Fixed Nodes",
        )
        ax.scatter(
            nodes[loaded_nodes, 0],
            nodes[loaded_nodes, 1],
            nodes[loaded_nodes, 2],
            color="green",
            label="Loaded Nodes",
        )

        # Label nodes with their index, shifted slightly to avoid overlap
        for i, (x, y, z) in enumerate(nodes):
            ax.text(
                x + 0.15,
                y + 0.15,
                z + 0.01,
                str(i),
                fontsize=12,
                ha="right",
                color="black",
            )

        ax.legend()
        plt.show()

    def animate_convergence(self, save_every_n):
        """Create an animation of the convergence process."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        elements = self.elements
        fixed = self.fixed
        external_forces = self.external_forces

        def update(frame):
            ax.clear()
            nodes = self.frames[frame]
            for edge in elements:
                ax.plot(*zip(*nodes[edge - 1]), color="black")

            loaded_nodes = np.any(external_forces != 0, axis=1)
            ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color="black")
            ax.scatter(
                nodes[np.array(fixed) - 1, 0],
                nodes[np.array(fixed) - 1, 1],
                nodes[np.array(fixed) - 1, 2],
                color="red",
                label="Fixed Nodes",
            )
            ax.scatter(
                nodes[loaded_nodes, 0],
                nodes[loaded_nodes, 1],
                nodes[loaded_nodes, 2],
                color="green",
            )

            for i, (x, y, z) in enumerate(nodes):
                ax.text(
                    x + 0.15,
                    y + 0.15,
                    z + 0.01,
                    str(i),
                    fontsize=12,
                    ha="right",
                    color="black",
                )

            # Calculate max displacement as the maximum change in node position
            if frame > 0:
                prev_nodes = self.frames[frame - 1]
                displacement = np.linalg.norm(nodes - prev_nodes, axis=1)
                max_displacement = np.max(displacement)
            else:
                max_displacement = 0.0

            # Display iteration number and max disp (in scientific notation)
            ax.text(
                0.1,
                3.5,
                0.1,
                f"Iteration: {frame * save_every_n}\n"
                f"Max Disp (m): {max_displacement:.3e}",
                fontsize=12,
                color="blue",
            )
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(-3, 1)

        _ = animation.FuncAnimation(
            fig, update, frames=len(self.frames), interval=50
        )
        plt.show()


# Example usage
# nodes, elements, external_loads, fixed_nodes = generate_struct(
#         5, spacing=2.5
#     )

nodes, elements, external_loads, fixed_nodes = generate_struct()

external_loads[5] = [0.0, 0.0, -1.5]  # Apply downward force to node 5

dr = DynamicRelaxation(
    nodes, elements, fixed_nodes, external_forces=external_loads
)

dr.plot_structure(dr.nodes)

nodes_new = dr.solve(save_every_n=10)

dr.plot_structure(
    nodes_new,
)

dr.animate_convergence(10)
