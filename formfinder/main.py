import importlib
import numpy as np
import scipy

import utils.log as hl  # helper functions for logging and debugging
import utils.solver as hs  # helper functions for solving linear systems
import utils.matrix as hm  # helper functions for creating structural matrices
import utils.plot as hp  # helper functions for plotting and visualization


class FormFinder:
    def __init__(
        self,
        solver="FD_fixed",
        structure="Schek_2D",
        debug=False,
        plot_save=False,
    ):
        self.solver = solver
        self.structure = structure
        self.debug = debug
        self.plot_save = plot_save

        # base output prefix
        self.out_path = f"./output/{self.solver}"

        hl.setup_logging(debug, solver)

        # dynamically import selected structure module
        struct_module = importlib.import_module(f"structures.{self.structure}")

        # Initialize structure, turn dicts --> arrays
        self.n, self.e, self.e_l, self.n_l, self.n_f = (
            hm.generate_struct_arrays(*struct_module.generate_struct())
        )

        if self.debug:
            hl.debug_struct_input(self.n, self.e, self.e_l, self.n_l, self.n_f)

        hp.plot_network_views(
            self.n,
            self.e,
            self.n_l,
            self.n_f,
            plot_text=True,
            save=self.plot_save,
            path=f"{self.out_path}_start.png",
        )

        self.initialize()

    def initialize(self):
        """Initialize structural matrices and parameters."""

        self.iteration = 0
        self.TOL = 1e-3
        self.MAX_ITER = 100

        # ------------------------------------
        #  Create Matrices
        # ------------------------------------
        self.C_total = hm.create_connectivity_matrix(self.n, self.e)

        self.L_vec, self.L, self.G = hm.create_length_matrix(
            self.n, self.C_total
        )

        self.p_x, self.p_y, self.p_z = hm.create_node_force_vectors(
            self.n_l, self.n_f
        )

        self.C_i, self.C_f = hm.partition_connectivity_matrix(
            self.C_total, self.n_f
        )

        if self.debug:
            hl.debug_struct_matrices(
                self.C_total,
                self.C_i,
                self.C_f,
                self.p_x,
                self.p_y,
                self.p_z,
                self.n_f,
            )

        # ------------------------------------
        #  Stacks for 3m x 3n SM operations
        # ------------------------------------
        self.C_i_3mn = hm.create_triple_stack(self.C_i)  # 3m x 3n (diagonal)
        self.C_f_3mn = hm.create_triple_stack(self.C_f)  # 3m x 3n (diagonal)
        self.GTxG = self.G[:, :, None] * self.G[:, None, :]  # m x 3x3
        self.I_3mn = np.broadcast_to(np.eye(3), self.GTxG.shape)

        # ------------------------------------
        #  Recorders
        # ------------------------------------
        self.node_pos_hist = [self.n.copy()]  # Store initial position
        self.L_total_hist = [np.sum(self.L_vec**2)]  # Init starting lens
        self.F_hist = []
        self.Q_hist = []

        self.done = False

        # ------------------------------------
        #  SM and DR parameters
        # ------------------------------------
        self.F_0 = np.diag(np.copy(self.e_l).flatten())

        # You can set these to whatever in normal analysis
        self.E = np.eye(len(self.e))
        self.A = np.eye(len(self.e))
        self.L_0 = np.copy(self.L)  # L_0 is initial len

        self.mode = "normal"

        match self.mode:
            case "normal":
                None
            case "constant_q":
                # Makes q = 1, when preload = 1
                self.E = np.eye(len(self.e))
                self.A = np.eye(len(self.e))
                self.L_0 = np.eye(self.L.shape[0])
            case "constant_f":
                # Will set f as whatever specified as preload
                self.L_0 = np.copy(self.L)  # L_0 is initial len

        # ------------------------------------
        #  DR parameters
        # ------------------------------------
        self.h = 0.1  # time-step
        self.gamma = 0.9  # damping
        self.v_x = np.zeros_like(self.p_x)
        self.v_y = np.zeros_like(self.p_y)
        self.v_z = np.zeros_like(self.p_z)

        self.KE_prev2 = 0.0
        self.KE_prev = 0.0
        self.KE_history = []
        self.first = True

    def solve(self):
        """Main loop for form finding."""

        for self.iteration in range(self.MAX_ITER):
            if self.debug:
                hl.debug_iteration(self.iteration, self.solver)

            if self.solver == "FD_linear":
                self.fd_linear_solver()
            elif self.solver == "FD_nonlinear":
                self.fd_nonlinear_solver()
            elif self.solver == "SM":
                self.sm_solver()
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
                    f"Iteration {self.iteration:03d}: "
                    f"Total Len = {self.L_total_hist[-1]:.3f}, "
                    f"Max error = {error:.3e}"
                )

                if self.debug:
                    hl.debug_new_nodes(self.n)
                    hl.debug_error(self.L_total_hist[-1], error)

            if error < self.TOL and self.KE_history[-1] < self.TOL:
                if self.done:  # If already flagged, exit completely
                    print("Final update complete. Exiting loop.")
                    break
                else:
                    print("Convergence achieved! Performing one final update.")
                    self.done = True  # Set flag for final update

        else:
            print("Max iterations reached without convergence.")

        self.post_process()

    def fd_linear_solver(self):
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

        if self.debug:
            hl.debug_F_L_Q(F, self.L, Q)
            hl.debug_stiffness_FD(K, D, D_f, self.n_f)

        self.d_x, self.d_y, self.d_z = hs.nodes_delta(
            self.p_x,
            self.p_y,
            self.p_z,
            K,
            D,
            D_f,
            *hm.partition_nodes_coordinates(self.n, self.n_f),
        )

        if self.debug:
            hl.debug_deltas(self.d_x, self.d_y, self.d_z, self.n_f)

        self.KE_history = [0]  # dont use in FD

    def fd_nonlinear_solver(self):
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

        if self.debug:
            hl.debug_F_L_Q(F, self.L, Q)
            hl.debug_stiffness_FD(K, D, D_f, self.n_f)

        self.d_x, self.d_y, self.d_z = hs.nodes_delta(
            self.p_x,
            self.p_y,
            self.p_z,
            K,
            D,
            D_f,
            *hm.partition_nodes_coordinates(self.n, self.n_f),
        )

        if self.debug:
            hl.debug_deltas(self.d_x, self.d_y, self.d_z, self.n_f)

        self.KE_history = [0]  # dont use in FD

    def sm_solver(self):
        """Stiffnes Method (SM) solver."""

        match self.mode:
            case "constant_f":
                self.L_0 = np.copy(self.L)

        F = hm.create_force_matrix(
            self.L,
            self.L_0,
            self.E,
            self.A,
            self.F_0,
        )
        Q = F @ np.linalg.inv(self.L)
        Q_3mn = hm.create_triple_stack(Q)  # (3m x 3m)

        self.F_hist.append(F)
        self.Q_hist.append(Q)

        if self.done:
            return

        # Geometric Stiffness (m x 3x3)
        k_g = np.diag(Q)
        k_g_3mn = k_g[:, None, None] * (self.I_3mn - self.GTxG)
        k_g_3mn_diag = scipy.linalg.block_diag(*k_g_3mn)  # diagonalize

        # Elastic Stiffness (m x 3x3)
        k_e = hm.create_elastic_k(self.E, self.A, self.L_0, as_matrix=False)
        k_e_3mn = k_e[:, None, None] * self.GTxG
        k_e_3mn_diag = scipy.linalg.block_diag(*k_e_3mn)  # diagonalize

        # Total Stiffness (3m x 3m, diagonal)
        K_total = k_g_3mn_diag + k_e_3mn_diag

        # Transformed stiffness, (3n_i x 3n_i)
        K = self.C_i_3mn.T @ K_total @ self.C_i_3mn

        # D and Df # (3n_i x 3n_i) (3n_i x 3n_f)
        D = self.C_i_3mn.T @ Q_3mn @ self.C_i_3mn
        D_f = self.C_i_3mn.T @ Q_3mn @ self.C_f_3mn

        if self.debug:
            hl.debug_F_L_Q(F, self.L, Q)
            hl.debug_stiffness_SM(
                k_g_3mn_diag, k_e_3mn_diag, K, D, D_f, self.n_f
            )

        self.d_x, self.d_y, self.d_z = hs.nodes_delta_3n(
            self.p_x,
            self.p_y,
            self.p_z,
            K,
            D,
            D_f,
            *hm.partition_nodes_coordinates(self.n, self.n_f),
        )

        if self.debug:
            hl.debug_deltas(self.d_x, self.d_y, self.d_z, self.n_f)

        self.KE_history = [0]  # dont use in SM

    def dr_implicit_solver(self):
        """Dynamic Relaxation (DR) solver."""

        match self.mode:
            case "constant_f":
                self.L_0 = np.copy(self.L)  # L_0 is initial len

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

        # Geometric and Elastic Stiffness (m x m)
        K_g = Q
        K_e = hm.create_elastic_k(self.E, self.A, self.L_0)
        K_total = K_g + K_e

        # Transformed Stiffness (n_i x n_i)
        K = self.C_i.T @ K_total @ self.C_i

        # # Kronecker delta as an identity matrix
        # # This seems to destabilize when doing leapfrog integration
        # delta = np.eye(K.shape[0])
        # K = K * delta

        # D and Df # (n_i x n_i) and (n_i x n_f)
        D = self.C_i.T @ Q @ self.C_i
        D_f = self.C_i.T @ Q @ self.C_f

        if self.debug:
            hl.debug_F_L_Q(F, self.L, Q)
            hl.debug_stiffness_DR(K_g, K_e, K, D, D_f, self.n_f)

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

        if self.debug:
            hl.debug_velocity_kinetic_energy(
                self.v_x, self.v_y, self.v_z, self.n_f, KE
            )
            hl.debug_deltas(self.d_x, self.d_y, self.d_z, self.n_f)

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

        # Geometric and Elastic Stiffness (m x m)
        K_g = Q
        K_e = hm.create_elastic_k(self.E, self.A, self.L_0)
        K_total = K_g + K_e

        # Transformed Stiffness (n_i x n_i)
        K = self.C_i.T @ K_total @ self.C_i
        # Kronecker delta as an identity matrix
        # Seems to destabilize sometimes when doing leapfrog
        # delta = np.eye(K.shape[0])
        # K = K * delta

        # D and Df # (n_i x n_i) and (n_i x n_f)
        D = self.C_i.T @ Q @ self.C_i
        D_f = self.C_i.T @ Q @ self.C_f

        if self.debug:
            hl.debug_F_L_Q(F, self.L, Q)
            hl.debug_stiffness_DR(K_g, K_e, K, D, D_f, self.n_f)

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

        if self.debug:
            hl.debug_velocity_kinetic_energy(
                self.v_x, self.v_y, self.v_z, self.n_f, KE
            )
            hl.debug_deltas(self.d_x, self.d_y, self.d_z, self.n_f)
            hl.debug_table([self.KE_prev2, self.KE_prev, KE])

        # Check for kinetic energy peak: KE_prev2 < KE_prev > KE
        if self.KE_prev2 < self.KE_prev > KE:

            q1 = (self.KE_prev - KE) / (
                (self.KE_prev - KE) - (self.KE_prev2 - self.KE_prev)
            )

            # this is same interp, different interval than the paper
            q2, KE_q, x_interp, y_interp = hs.quadratic_interp(
                [self.KE_prev2, self.KE_prev, KE]
            )
            hp.plot_quadratic_interp(
                [0, 0.5, 1.0],
                [self.KE_prev2, self.KE_prev, KE],
                x_interp,
                y_interp,
                q1,
                q2,
                KE_q,
                t=self.iteration,
                save=self.plot_save,
                path=f"{self.out_path}_interp_{self.iteration}.png",
            )

            q = q1

            if self.debug:
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

            if self.debug:
                hl.debug_deltas(self.d_x, self.d_y, self.d_z, self.n_f)

            # Reset velocities to 0 (kinetic damping)
            self.first = True

        self.KE_prev2 = self.KE_prev
        self.KE_prev = KE
        self.KE_history.append(KE)

    def post_process(self):
        """Final visualization and debugging."""
        hp.plot_kinetic_energy(
            self.KE_history,
            self.solver,
            save=self.plot_save,
            path=f"{self.out_path}_kinetic_energy.png",
        )

        hp.plot_animation(
            self.node_pos_hist,
            self.e,
            self.n_f,
            t=200,
            z_scale=1.0,
            plot_text=False,
            save=self.plot_save,
            path=f"{self.out_path}_animation.gif",
        )

        hp.plot_network_views(
            self.n,
            self.e,
            self.n_l,
            self.n_f,
            plot_text=False,
            save=self.plot_save,
            path=f"{self.out_path}_final.png",
        )

        if self.debug:
            hl.debug_final(
                self.n,
                self.F_hist[-1],
                self.L,  # it's been updating
                self.Q_hist[-1],
            )


if __name__ == "__main__":

    solvers = {
        1: "FD_linear",
        2: "FD_nonlinear",
        3: "SM",
        4: "DR_imp",
        5: "DR_leap",
    }

    structs = {
        1: "Schek_2D",
        2: "Schek_Modified",
        3: "Veenendaal",
        4: "Pauletti",
        5: "CEE6501_2D",
        6: "CEE6501_PointLoad",
        7: "Star",
        8: "Test_Constrained",
    }

    run = FormFinder(
        solver=solvers[2], structure=structs[1], debug=True, plot_save=True
    )

    run.solve()
