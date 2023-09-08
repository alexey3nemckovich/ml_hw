import logging
from constants import *


def create_logger() -> logging.Logger:
    """
    create_logger returns default configured NLP module logger.

    Example usage:
        create_logger()
    """
    logger_instance = logging.getLogger(NLP_LOGGER_NAME)

    logger_instance.setLevel(logging.INFO)

    file_handler = logging.FileHandler(NLP_LOG)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(LOGGER_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger_instance.addHandler(file_handler)
    logger_instance.addHandler(console_handler)

    return logger_instance


logger = create_logger()
