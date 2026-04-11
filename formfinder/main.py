import numpy as np
import scipy

import utils.log as hl
import utils.solver as hs
import utils.matrix as hm
import utils.plot as hp

from structures.struct_2 import generate_struct


class FormFinder:
    def __init__(self, solver="FD_fixed", debug=False):
        self.solver = solver
        self.debug = debug
        hl.setup_logging(debug, solver)

        # Initialize structure, turn dicts --> arrays
        self.n, self.e, self.e_l, self.n_l, self.n_f = (
            hm.generate_struct_arrays(*generate_struct())
        )

        hl.debug_struct_input(self.n, self.e, self.e_l, self.n_l, self.n_f)

        hp.plot_network_views(
            self.n, self.e, self.n_l, self.n_f, plot_text=True
        )

        self.initialize()

    def initialize(self):
        """Initialize structural matrices and parameters."""

        self.C_total = hm.create_connectivity_matrix(self.n, self.e)

        self.L_vec, self.L, self.G = hm.create_length_matrix(
            self.n, self.C_total
        )

        self.C_i, self.C_f = hm.partition_connectivity_matrix(
            self.C_total, self.n_f
        )

        self.p_x, self.p_y, self.p_z = hm.create_node_force_vectors(
            self.n_l, self.n_f
        )

        hl.debug_struct_matrices(
            self.C_total, self.C_i, self.C_f, self.p_x, self.p_y, self.p_z
        )

        # Recorders
        self.node_pos_hist = [self.n.copy()]  # Store initial position
        self.L_total_hist = [np.sum(self.L_vec**2)]
        self.F_hist = []
        self.Q_hist = []

        self.done = False

        # Dynamic Relaxation parameters
        # self.L_0 = np.copy(self.L)
        self.L_0 = np.eye(self.L.shape[0])  # for Struct_2, benchmark
        self.F_0 = np.diag(np.copy(self.e_l).flatten())
        self.E = np.eye(len(self.e))
        self.A = np.eye(len(self.e))
        # self.E = np.diag(np.zeros(len(self.e)))
        # self.A = np.diag(np.zeros(len(self.e)))
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
        """Main loop for form finding."""
        TOL = 1e-3
        MAX_ITER = 100

        for iteration in range(MAX_ITER):
            hl.debug_iteration(iteration, self.solver)

            if self.solver == "FD_fixed":
                self.fd_fixed_solver()
            elif self.solver == "FD_iter":
                self.fd_iter_solver()
            elif self.solver == "SM":
                self.sm()
            elif self.solver == "DR_imp":
                self.dr_implicit_solver()
            elif self.solver == "DR_leap":
                self.dr_leapfrog_solver()
            else:
                raise ValueError(f"Unknown solver type: {self.solver}")

            if not self.done:
                # Update nodes
                self.n = hs.nodes_update(
                    self.n, self.d_x, self.d_y, self.d_z, self.n_f
                )
                self.node_pos_hist.append(self.n.copy())

                # Check for convergence
                self.L_vec, self.L, self.G = hm.create_length_matrix(
                    self.n, self.C_total
                )
                self.L_total_hist.append(np.sum(self.L_vec**2))
                error = np.abs(self.L_total_hist[-2] - self.L_total_hist[-1])

                print(
                    f"Iteration {iteration}: "
                    f"Total Len = {self.L_total_hist[-1]:.3f}, "
                    f"Max error = {error:.3e}"
                )

                hl.debug_new_nodes(self.n)
                hl.debug_error(self.L_total_hist[-1], error)

            if error < TOL and self.KE_history[-1] < TOL:
                if self.done:  # If already flagged, exit completely
                    print("Final update complete. Exiting loop.")
                    break
                else:
                    print("Convergence achieved! Performing one final update.")
                    self.done = True  # Set flag for final update

        else:
            print("Max iterations reached without convergence.")

        self.post_process()

    def fd_fixed_solver(self):
        """Force Density (FD) solver with fixed Q."""
        Q = np.diag(self.e_l.flatten())
        F = Q @ self.L
        K = self.C_i.T @ Q @ self.C_i
        D = self.C_i.T @ Q @ self.C_i
        D_f = self.C_i.T @ Q @ self.C_f

        self.F_hist.append(F)
        self.Q_hist.append(Q)

        if self.done:
            return

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
        self.KE_history = [0]  # dont use in FD

    def fd_iter_solver(self):
        """Force Density (FD) solver with fixed F."""
        F = np.diag(self.e_l.flatten())
        Q = F @ np.linalg.inv(self.L)
        K = self.C_i.T @ Q @ self.C_i
        D = self.C_i.T @ Q @ self.C_i
        D_f = self.C_i.T @ Q @ self.C_f

        self.F_hist.append(F)
        self.Q_hist.append(Q)

        if self.done:
            return

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
        self.KE_history = [0]  # dont use in FD

    def sm(self):
        """Stiffnes Method (SM) solver."""
        F = hm.create_force_matrix(
            self.L,
            self.L_0,
            self.E,
            self.A,
            self.F_0,
        )
        Q = F @ np.linalg.inv(self.L)
        self.F_hist.append(F)
        self.Q_hist.append(Q)

        if self.done:
            return

        hl.debug_force_and_density(F, Q)

        # C 3x3
        C_i_3x3 = hm.create_triple_stack(self.C_i)
        C_f_3x3 = hm.create_triple_stack(self.C_f)

        # creates and m x [3x3] matrix
        GTG = self.G[:, :, None] * self.G[:, None, :]

        # Elastic Stiffness
        K_e = hm.create_elastic_stiffness_matrix(self.E, self.A, self.L_0)
        k_e = np.diag(K_e)
        k_e_3x3 = k_e[:, None, None] * GTG

        # Geometric Stiffness
        K_g = Q
        k_g = np.diag(K_g)
        I_3x3 = np.broadcast_to(np.eye(3), GTG.shape)
        k_g_3x3 = k_g[:, None, None] * (I_3x3 - GTG)

        K_total = k_e_3x3 + k_g_3x3

        K_total_diag = scipy.linalg.block_diag(*K_total)

        K = C_i_3x3.T @ K_total_diag @ C_i_3x3

        ####
        Q_3x3 = hm.create_triple_stack(Q)

        D = C_i_3x3.T @ Q_3x3 @ C_i_3x3
        D_f = C_i_3x3.T @ Q_3x3 @ C_f_3x3

        self.d_x, self.d_y, self.d_z = hs.nodes_delta2(
            self.p_x,
            self.p_y,
            self.p_z,
            K,
            D,
            D_f,
            *hm.partition_nodes_coordinates(self.n, self.n_f),
        )

        hl.debug_deltas(self.d_x, self.d_y, self.d_z)
        self.KE_history = [0]  # dont use in FD

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
        self.F_hist.append(F)
        self.Q_hist.append(Q)

        if self.done:
            return

        hl.debug_force_and_density(F, Q)

        K_g = Q
        K_e = hm.create_elastic_stiffness_matrix(self.E, self.A, self.L_0)
        K_total = K_g + K_e

        K = self.C_i.T @ K_total @ self.C_i
        # Kronecker delta as an identity matrix
        # This seems to destabilize when doing leapfrog integration
        # delta = np.eye(K.shape[0])
        # K = K * delta

        hl.debug_stiffness(K_g, K_e, K)

        D = self.C_i.T @ Q @ self.C_i
        D_f = self.C_i.T @ Q @ self.C_f

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

        self.F_hist.append(F)
        self.Q_hist.append(Q)

        if self.done:
            return

        hl.debug_force_and_density(F, Q)

        K_g = Q
        K_e = hm.create_elastic_stiffness_matrix(self.E, self.A, self.L_0)
        K_total = K_g + K_e

        K = self.C_i.T @ K_total @ self.C_i

        # Kronecker delta as an identity matrix
        # Seems to destabilize sometimes when doing leapfrog
        # delta = np.eye(K.shape[0])
        # K = K * delta

        hl.debug_stiffness(K_g, K_e, K)

        D = self.C_i.T @ Q @ self.C_i
        D_f = self.C_i.T @ Q @ self.C_f

        self.d_x, self.d_y, self.d_z = hs.nodes_delta(
            self.p_x,
            self.p_y,
            self.p_z,
            K,
            D,
            D_f,
            *hm.partition_nodes_coordinates(self.n, self.n_f),
        )

        # For use in energy peak backtrack
        d_x_save = np.copy(self.d_x)
        d_y_save = np.copy(self.d_y)
        d_z_save = np.copy(self.d_z)

        # M = h^2/2 * K, V1 = V0 + h/M * f (normal)
        # M = h^2/2 * K, V1 = h/2*M * f (first iteration)
        if self.first:
            self.v_x = self.gamma * self.h * (1 / self.h**2) * self.d_x
            self.v_y = self.gamma * self.h * (1 / self.h**2) * self.d_y
            self.v_z = self.gamma * self.h * (1 / self.h**2) * self.d_z
            self.first = False
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

        # Check for kinetic energy peak: KE_prev2 < KE_prev > KE
        if self.KE_prev2 < self.KE_prev > KE:

            q1 = (self.KE_prev - KE) / (
                (self.KE_prev - KE) - (self.KE_prev2 - self.KE_prev)
            )

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

            self.d_x -= (
                self.h * (1 + q) * self.gamma * self.v_x
                + self.gamma * q * d_x_save
            )
            self.d_y -= (
                self.h * (1 + q) * self.gamma * self.v_y
                + self.gamma * q * d_y_save
            )
            self.d_z -= (
                self.h * (1 + q) * self.gamma * self.v_z
                + self.gamma * q * d_z_save
            )

            hl.debug_deltas(self.d_x, self.d_y, self.d_z)

            # Reset velocities to 0 (kinetic damping)
            self.first = True

        self.KE_prev2 = self.KE_prev
        self.KE_prev = KE
        self.KE_history.append(KE)

    def post_process(self):
        """Final visualization and debugging."""
        hp.plot_kinetic_energy(self.KE_history, self.solver)
        hp.plot_animation(
            self.node_pos_hist,
            self.e,
            self.n_f,
            t=1,
            plot_text=False,
            save_gif=True,
        )
        hp.plot_network_views(
            self.n, self.e, self.n_l, self.n_f, plot_text=True
        )
        hl.debug_final(
            self.n,
            self.L,
            self.F_hist[-1],
            self.Q_hist[-1],
        )


if __name__ == "__main__":
    # simulation = FormFinder(solver="FD_fixed", debug=True)
    # simulation = FormFinder(solver="FD_iter", debug=True)
    # simulation = FormFinder(solver="DR_imp", debug=True)
    # simulation = FormFinder(solver="DR_leap", debug=True)
    simulation = FormFinder(solver="SM", debug=True)
    simulation.solve()
