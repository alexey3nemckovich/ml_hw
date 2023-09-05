import pandas as pd
import os


def read_csv(filepath: str, dropna: bool = True, encoding: str = 'latin1'):
    data = pd.read_csv(filepath, encoding=encoding)
    if dropna:
        data = data.dropna()
    return data
    #return data[:10]


def save_int_to_file(filepath: str, number: int):
    with open(filepath, "w") as file:
        file.write(str(number))


def read_int_from_file(filepath: str):
    with open(filepath, "r") as file:
        return int(file.read())


def file_exists(filepath: str):
    return os.path.exists(filepath)
