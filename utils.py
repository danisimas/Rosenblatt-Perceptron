import numpy as np


def read_data(filename: str):
    # Read data from file
    data = np.fromfile(f"./task/data/{filename}.txt")

    # Separate values into (x1 x2 y) elements
    data = data.reshape(-1, 3)

    # Reorganize valkues into [([x1.1,x1.2], y1), ([x2.1,x2.2], y2)]
    data = np.array([(row[:2], row[2]) for row in data], dtype=object)

    return data


def identifilier(dict):
    pass
