import logging


# Set up logging
def setup_logging(debug=False):
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if debug:
        logging.basicConfig(
            filename="./debug_log.txt",
            level=logging.DEBUG,
            format="%(message)s",
            filemode="w",
        )
    else:
        logging.basicConfig(level=logging.INFO)


def debug_initial_struct(
    n, e, e_l, n_l, n_f, connectivity_matrix, C, C_f, p_x, p_y, p_z
):
    logging.debug("\nNodes:\n %s", n)
    logging.debug("\nElements:\n %s", e)
    logging.debug("\nElements (Preload):\n %s", e_l)
    logging.debug("\nNodes (Loads):\n %s", n_l)
    logging.debug("\nNodes (Fixed):\n %s", n_f)

    logging.debug("\nConnectivity Matrix:\n %s", connectivity_matrix)
    logging.debug("\nC (free nodes):\n %s", C)
    logging.debug("\nCf (fixed nodes):\n %s", C_f)
    logging.debug("\np_x:\n %s", p_x)
    logging.debug("\np_y:\n %s", p_y)
    logging.debug("\np_z:\n %s", p_z)


def debug_iteration(iteration, solver):
    logging.debug("")
    logging.debug("######################")
    logging.debug("# ITERATION: %s", iteration)
    logging.debug("# SOLVER: %s", solver)
    logging.debug("######################")


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

    logging.debug("\n%s", table)
    return before
