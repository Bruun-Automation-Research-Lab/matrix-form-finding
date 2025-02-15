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
