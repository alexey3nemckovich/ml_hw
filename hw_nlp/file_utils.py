import os
import pandas as pd


def read_csv(filepath: str, dropna: bool = True, encoding: str = 'latin1') -> pd.DataFrame:
    """
    read_csv returns DataFrame read from specified CSV format file.

    :param filepath: A path to file with data.
    :param dropna: Flag specifying if missed values should be deleted or not.
    :param encoding: Encoding to use for UTF when reading.
    :returns: A Pandas DataFrame.

    Example usage:
        read_csv(filepath='./data.csv', dropna=False)
    """
    data = pd.read_csv(filepath, encoding=encoding)
    if dropna:
        data = data.dropna()
    return data
    # return data[:10]


def save_int_to_file(filepath: str, number: int):
    """
    save_int_to_file saves int number to the specified file.

    :param filepath: A path to file that will be created/overwritten.
    :param number: Int number to save.

    Example usage:
        save_int_to_file(filepath='./data.txt', number=7)
    """
    with open(filepath, "w") as file:
        file.write(str(number))


def read_int_from_file(filepath: str) -> int:
    """
    read_int_from_file reads int from file.

    :param filepath: A path to file with data.
    :returns: Int number read from file.

    Example usage:
        read_int_from_file(filepath='./data.txt')
    """
    with open(filepath, "r") as file:
        return int(file.read())


def file_exists(filepath: str) -> bool:
    """
    read_int_from_file returns flag specifying whether file exists.

    :param filepath: A path to file with data.
    :returns: Int number read from file.

    Example usage:
        file_exists(filepath='./data.txt')
    """
    return os.path.exists(filepath)
