import numpy as np

import helper_log as hl
import helper_solver as hs
import helper_matrix as hm
import helper_plot as hp

from structures.struct_3 import generate_struct


class FormFinder:
    def __init__(self, solver="FD_fixed", debug=False):
        self.solver = solver
        self.debug = debug
        hl.setup_logging(debug)

        # Initialize structure, turn dicts --> arrays
        self.n, self.e, self.e_l, self.n_l, self.n_f = (
            hm.generate_struct_arrays(*generate_struct())
        )

        hl.debug_struct_input(self.n, self.e, self.e_l, self.n_l, self.n_f)

        self.initialize()

    def initialize(self):
        """Initialize structural matrices and parameters."""

        self.L_vec, self.L = hm.create_length_matrix(self.n, self.e)

        C_total = hm.create_connectivity_matrix(self.n, self.e)
        self.C, self.C_f = hm.partition_connectivity_matrix(C_total, self.n_f)

        self.p_x, self.p_y, self.p_z = hm.create_node_force_vectors(
            self.n_l, self.n_f
        )

        hl.debug_struct_matrices(
            C_total, self.C, self.C_f, self.p_x, self.p_y, self.p_z
        )

        # Recorders
        self.node_pos_hist = [self.n.copy()]  # Store initial position
        self.L_total_hist = [np.sum(self.L_vec**2)]
        self.F_hist = []
        self.Q_hist = []

        # Dynamic Relaxation parameters
        self.L_0 = np.copy(self.L)
        self.F_0 = np.diag(np.copy(self.e_l).flatten())
        self.E = np.eye(len(self.e))
        self.A = np.eye(len(self.e))
        self.h = 0.1
        self.gamma = 1.0
        self.v_x = np.zeros_like(self.p_x)
        self.v_y = np.zeros_like(self.p_y)
        self.v_z = np.zeros_like(self.p_z)

        self.KE_prev2 = 0.0
        self.KE_prev = 0.0
        self.KE_history = []
        self.first = True

    def solve(self):
        """Main loop for structural simulation."""
        TOL = 1e-4
        MAX_ITER = 10000

        for iteration in range(MAX_ITER):
            hl.debug_iteration(iteration, self.solver)

            if self.solver == "FD_fixed":
                self.fd_fixed_solver()
            elif self.solver == "FD_iter":
                self.fd_iter_solver()
            elif self.solver == "DR_imp":
                self.dr_implicit_solver()
            elif self.solver == "DR_leap":
                self.dr_leapfrog_solver()
            else:
                raise ValueError(f"Unknown solver type: {self.solver}")

            # Update nodes
            self.n = hs.nodes_update(
                self.n, self.d_x, self.d_y, self.d_z, self.n_f
            )
            self.node_pos_hist.append(self.n.copy())

            # Check for convergence
            self.L_vec, self.L = hm.create_length_matrix(self.n, self.e)
            self.L_total_hist.append(np.sum(self.L_vec**2))
            error = np.abs(self.L_total_hist[-2] - self.L_total_hist[-1])

            print(
                f"Iteration {iteration + 1}: "
                f"Total Len = {self.L_total_hist[-1]:.3f}, "
                f"Max error = {error:.3e}"
            )

            hl.debug_new_nodes(self.n)
            hl.debug_error(self.L_total_hist[-1], error)

            if error < TOL:
                print("Convergence achieved!")
                break

        else:
            print("Max iterations reached without convergence.")

        self.post_process()

    def fd_fixed_solver(self):
        """Force Density (FD) solver with fixed Q."""
        Q = np.diag(self.e_l.flatten())
        F = Q @ self.L
        K = self.C.T @ Q @ self.C
        D = self.C.T @ Q @ self.C
        D_f = self.C.T @ Q @ self.C_f

        hl.debug_force_and_density(F, Q)
        hl.debug_stiffness_FD(K, D, D_f)

        self.d_x, self.d_y, self.d_z = hs.nodes_delta(
            self.p_x,
            self.p_y,
            self.p_z,
            K,
            D,
            D_f,
            *hm.partition_nodes_coordinates(self.n, self.n_f),
        )

        hl.debug_deltas(self.d_x, self.d_y, self.d_z)

        self.F_hist.append(F)
        self.Q_hist.append(Q)

    def fd_iter_solver(self):
        """Force Density (FD) solver with fixed F."""
        F = np.diag(self.e_l.flatten())
        Q = F @ np.linalg.inv(self.L)
        K = self.C.T @ Q @ self.C
        D = self.C.T @ Q @ self.C
        D_f = self.C.T @ Q @ self.C_f

        hl.debug_force_and_density(F, Q)
        hl.debug_stiffness_FD(K, D, D_f)

        self.d_x, self.d_y, self.d_z = hs.nodes_delta(
            self.p_x,
            self.p_y,
            self.p_z,
            K,
            D,
            D_f,
            *hm.partition_nodes_coordinates(self.n, self.n_f),
        )

        hl.debug_deltas(self.d_x, self.d_y, self.d_z)

        self.F_hist.append(F)
        self.Q_hist.append(Q)

    def dr_implicit_solver(self):
        """Dynamic Relaxation (DR) solver."""
        F = hm.create_force_matrix(
            self.L,
            self.L_0,
            self.E,
            self.A,
            self.F_0,
        )
        Q = F @ np.linalg.inv(self.L)
        hl.debug_force_and_density(F, Q)

        K_g = Q
        K_e = hm.create_elastic_stiffness_matrix(self.E, self.A, self.L_0)
        K_total = K_g + K_e

        K = self.C.T @ K_total @ self.C
        # Kronecker delta as an identity matrix
        # This seems to destabilize when doing leapfrog integration
        # delta = np.eye(K.shape[0])
        # K = K * delta

        hl.debug_stiffness(K_g, K_e, K)

        D = self.C.T @ Q @ self.C
        D_f = self.C.T @ Q @ self.C_f

        self.d_x, self.d_y, self.d_z = hs.nodes_delta(
            self.p_x,
            self.p_y,
            self.p_z,
            K,
            D,
            D_f,
            *hm.partition_nodes_coordinates(self.n, self.n_f),
        )

        self.v_x = self.gamma * self.h * (2 / self.h**2) * self.d_x
        self.v_y = self.gamma * self.h * (2 / self.h**2) * self.d_y
        self.v_z = self.gamma * self.h * (2 / self.h**2) * self.d_z

        self.d_x = self.v_x * self.h
        self.d_y = self.v_y * self.h
        self.d_z = self.v_z * self.h

        KE = hs.compute_kinetic_energy(K, self.v_x, self.v_y, self.v_z, self.h)
        self.KE_history.append(KE)

        hl.debug_velocity_kinetic_energy(self.v_x, self.v_y, self.v_z, KE)
        hl.debug_deltas(self.d_x, self.d_y, self.d_z)

        self.F_hist.append(F)
        self.Q_hist.append(Q)

    def dr_leapfrog_solver(self):
        """Dynamic Relaxation (DR) solver."""
        F = hm.create_force_matrix(
            self.L,
            self.L_0,
            self.E,
            self.A,
            self.F_0,
        )
        Q = F @ np.linalg.inv(self.L)
        hl.debug_force_and_density(F, Q)

        K_g = Q
        K_e = hm.create_elastic_stiffness_matrix(self.E, self.A, self.L_0)
        K_total = K_g + K_e

        K = self.C.T @ K_total @ self.C
        # Kronecker delta as an identity matrix
        # This seems to destabilize when doing leapfrog integration
        # delta = np.eye(K.shape[0])
        # K = K * delta

        hl.debug_stiffness(K_g, K_e, K)

        D = self.C.T @ Q @ self.C
        D_f = self.C.T @ Q @ self.C_f

        self.d_x, self.d_y, self.d_z = hs.nodes_delta(
            self.p_x,
            self.p_y,
            self.p_z,
            K,
            D,
            D_f,
            *hm.partition_nodes_coordinates(self.n, self.n_f),
        )

        # M = h^2/2 * K, V1 = V0 + h/M * f (normal)
        # M = h^2/2 * K, V1 = h/2*M * f (first iteration)
        if self.first:
            self.v_x = self.gamma * self.h * (1 / self.h**2) * self.d_x
            self.v_y = self.gamma * self.h * (1 / self.h**2) * self.d_y
            self.v_z = self.gamma * self.h * (1 / self.h**2) * self.d_z
            self.first = True
        else:
            self.v_x += self.gamma * self.h * (2 / self.h**2) * self.d_x
            self.v_y += self.gamma * self.h * (2 / self.h**2) * self.d_y
            self.v_z += self.gamma * self.h * (2 / self.h**2) * self.d_z

        self.d_x = self.v_x * self.h
        self.d_y = self.v_y * self.h
        self.d_z = self.v_z * self.h

        KE = hs.compute_kinetic_energy(K, self.v_x, self.v_y, self.v_z, self.h)

        hl.debug_velocity_kinetic_energy(self.v_x, self.v_y, self.v_z, KE)
        hl.debug_deltas(self.d_x, self.d_y, self.d_z)
        hl.debug_table([self.KE_prev2, self.KE_prev, KE])

        self.KE_prev2 = self.KE_prev
        self.KE_prev = KE
        self.KE_history.append(KE)

        self.F_hist.append(F)
        self.Q_hist.append(Q)

    def post_process(self):
        """Final visualization and debugging."""
        hp.plot_kinetic_energy(self.KE_history, self.solver)
        hp.plot_animation(self.node_pos_hist, self.e, self.n_f, t=1)
        hp.plot_network_views(self.n, self.e, self.n_l, self.n_f)
        hl.debug_final(
            self.n,
            self.L,
            self.F_hist[-1],
            self.Q_hist[-1],
        )


if __name__ == "__main__":
    # simulation = FormFinder(solver="FD_iter", debug=True)
    simulation = FormFinder(solver="DR_imp", debug=True)
    simulation.solve()
