import logging as log
import numpy as np


# Set up log
def setup_logging(debug=False):
    log.getLogger("matplotlib").setLevel(log.WARNING)

    if debug:
        log.basicConfig(
            filename="./debug_log.txt",
            level=log.DEBUG,
            format="%(message)s",
            filemode="w",
        )
    else:
        log.basicConfig(level=log.INFO)


def debug_initial_struct(
    n, e, e_l, n_l, n_f, connectivity_matrix, C, C_f, p_x, p_y, p_z
):

    log.debug("######################")
    log.debug("# INPUT STRUCTURE")
    log.debug("######################")

    log.debug("\nNodes:\n %s", n)
    log.debug("\nElements:\n %s", e)
    log.debug("\nElements (Preload):\n %s", e_l)
    log.debug("\nNodes (Loads):\n %s", n_l)
    log.debug("\nNodes (Fixed):\n %s", n_f)

    log.debug("")
    log.debug("#-------------------------")
    log.debug("\np_x:\n %s", p_x)
    log.debug("\np_y:\n %s", p_y)
    log.debug("\np_z:\n %s", p_z)

    log.debug("")
    log.debug("#-------------------------")
    log.debug("\nConnectivity Matrix:\n %s", connectivity_matrix)
    log.debug("\nC (free n), m x n_free:\n %s", C)
    log.debug("\nCf (fixed n), m x n_fixed:\n %s", C_f)


def debug_iteration(iteration, solver):
    log.debug("")
    log.debug("######################")
    log.debug("# ITERATION: %s", iteration)
    log.debug("# SOLVER: %s", solver)
    log.debug("######################")


def debug_force_and_density(F, Q):
    log.debug("")
    log.debug("#-------------------------")
    log.debug("\nForce, [m x m] -> [m x 1]:\n %s", np.diag(F))
    log.debug("\nForce Density, [m x m] -> [m x 1]:\n %s", np.diag(Q))


def debug_stiffness(K_g, K_e, K, K_mod):
    log.debug("")
    log.debug("#-------------------------")
    log.debug("\nElement K_g, diag:\n %s", np.diag(K_g))
    log.debug("\nElement K_e, diag:\n %s", np.diag(K_e))

    log.debug("\nK (free nodes):\n %s", K)
    log.debug("\nK (free nodes), diag:\n %s", np.diag(K_mod))


def debug_stiffness_FD(K, D, D_f):
    log.debug("")
    log.debug("#-------------------------")
    log.debug("\nK, [n_free x n_free]:\n %s", K)
    log.debug("\nD, [n_free x n_free]:\n %s", D)
    log.debug("\nD_f, [n_free x n_fixed]:\n %s", D_f)


def debug_deltas(d_x, d_y, d_z):
    log.debug("")
    log.debug("#-------------------------")
    log.debug("\nd_x [n_free x 1]:\n %s", d_x)
    log.debug("\nd_y [n_free x 1]:\n %s", d_y)
    log.debug("\nd_z [n_free x 1]:\n %s", d_z)


def debug_new_nodes(n_new):
    log.debug("")
    log.debug("#-------------------------")
    log.debug("\nnodes new []:\n %s\n", n_new)


def debug_velocity_kinetic_energy(v_x, v_y, v_z, KE):
    log.debug("")
    log.debug("#-------------------------")
    log.debug("\nv_x:\n %s", v_x)
    log.debug("\nv_y:\n %s", v_y)
    log.debug("\nv_z:\n %s", v_z)
    log.debug("")
    log.debug("\nKINETIC ENERGY: %s", KE)


def debug_energy_peak(q):
    log.debug("")
    log.debug("\nKINETIC ENERGY PEAK REACHED. APPLYING DAMPING.")
    log.debug(f"Interpolated q = {q:.3f}")


def debug_error(L_total, error):
    log.debug("")
    log.debug("#-------------------------")
    log.debug(f"Total Len = {L_total:.3f}, ")
    log.debug(f"Max error = {error:.3f}")


def debug_final(n_new, L_new, F, Q):
    log.debug("")
    log.debug("######################")
    log.debug("# FINAL STRUCTURE")
    log.debug("######################")
    log.debug("\nFinal Nodes:\n %s", n_new)
    log.debug("\nFinal Element Lengths:\n %s", np.diag(L_new))
    log.debug("\nFinal Element Forces:\n %s", np.diag(F))
    log.debug("\nFinal Element Force Densities:\n %s", np.diag(Q))
    log.debug(
        "\nFinal Element Forces (f/f_avg):\n %s",
        np.diag(F) / np.average(np.diag(F)),
    )


def debug_table2(r1):
    """
    This function generates a debug table

    Args:
    - y_star (float): The value of t* (time).
    - E_star (float): The energy at t* (E(t*)).
    - energy_values (list of floats): List containing the other energy values.
    """

    # Ensure energy_values has exactly 3 points
    if len(r1) != 3:
        raise ValueError("energy_values should contain exactly 3 points.")

    # Prepare the ASCII table with reordered columns
    table = f"""
    +----+------------+-----------+-----------+
    | t  | t - 3dt/2  | t - dt/2  | t + dt/2  |
    +----+------------+-----------+-----------+
    | E  | {r1[0]:.3e}  | {r1[1]:.3e} | {r1[2]:.3e} |
    +----+------------+-----------+-----------+
    """

    log.debug("\n%s", table)


def debug_table(y_star, E_star, energy_values):
    """
    This function generates a debug table

    Args:
    - y_star (float): The value of t* (time).
    - E_star (float): The energy at t* (E(t*)).
    - energy_values (list of floats): List containing the other energy values.
    """

    # Ensure energy_values has exactly 3 points
    if len(energy_values) != 3:
        raise ValueError("energy_values should contain exactly 3 points.")

    # Build the time array based on the value of y_star
    if 0 <= y_star < 0.5:
        r1 = [0.0, y_star, 0.5, 1.0]
        r2 = [energy_values[0], E_star, energy_values[1], energy_values[2]]
        before = True
    elif 0.5 <= y_star <= 1.0:
        r1 = [0.0, 0.5, y_star, 1.0]
        r2 = [energy_values[0], energy_values[1], E_star, energy_values[2]]
        before = False

    # Prepare the ASCII table with reordered columns
    table = f"""
    +----+------------+-----------+-----------+------------+
    | t  | {r1[0]:.3f}      | {r1[1]:.3f}     | {r1[2]:.3f}     | {r1[3]:.3f}
    +----+------------+-----------+-----------+------------+
    | E  | {r2[0]:.3e} | {r2[1]:.3e} | {r2[2]:.3e} | {r2[3]:.3e}
    +----+------------+-----------+-----------+------------+
    """

    log.debug("\n%s", table)
    return before
