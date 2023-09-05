import logging
from constants import *


def create_logger():
    logger = logging.getLogger(NLP_LOGGER_NAME)

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(NLP_LOG)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(LOGGER_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = create_logger()
