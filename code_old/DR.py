import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from structures.struct_4 import generate_struct

from helper_matrix import (
    generate_struct_arrays,
)

from helper_plot import (
    plot_animation,
    plot_network_views,
)


class DynamicRelaxation:
    def __init__(
        self,
        nodes,
        elements,
        fixed,
        mass=1.0,
        damping=0.70,
        dt=0.1,
        tol=1e-3,
        external_forces=None,
    ):
        # Convert the nodes dictionary into a NumPy array
        self.nodes = nodes  # Node coordinates
        self.elements = elements  # connectivity
        self.fixed = fixed  # Fixed nodes (list of node indices)
        self.mass = mass  # Mass per node
        self.damping = damping  # Damping factor
        self.dt = dt  # Time step
        self.tol = tol  # Convergence tolerance
        self.velocities = np.zeros_like(self.nodes)  # Initial velocities
        # Convert external forces dictionary into a NumPy array
        self.external_forces = external_forces
        self.frames = []  # Store frames for animation

    def length(self, edge):
        """Compute length of an edge."""
        return np.linalg.norm(
            self.nodes[edge[1] - 1] - self.nodes[edge[0] - 1]
        )

    def solve(self, target_length=None):
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
            forces[self.fixed.flatten() == 1] = 0

            # Update velocities and positions
            self.velocities += (forces / self.mass) * self.dt
            self.velocities *= self.damping
            self.nodes += self.velocities * self.dt

            # Store frames for animation
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


nodes, elements, elements_preload, nodes_load, fixed_nodes = generate_struct()

n, e, e_l, n_l, n_f = generate_struct_arrays(
    nodes, elements, elements_preload, nodes_load, fixed_nodes
)

dr = DynamicRelaxation(n, e, n_f, external_forces=n_l)

plot_network_views(n, e, n_l, n_f)

nodes_new = dr.solve()

plot_network_views(dr.nodes, dr.elements, dr.external_forces, dr.fixed)

plot_animation(dr.frames, e, n_f)
