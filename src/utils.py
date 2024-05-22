import numpy as np


def read_data(filename: str):
    # Read data from file
    data = np.fromfile(f"./task/data/data{filename}.txt")

    # Separate values into (x1 x2 y) elements
    data = data.reshape(-1, 3)

    # Reorganize valkues into [([x1.1,x1.2], y1), ([x2.1,x2.2], y2)]
    data = np.array([(row[:2], row[2]) for row in data], dtype=object)

    return data


def identifier(values):
    # Pick last digit of each value
    last_digits = (int(value[-1]) for value in values)

    # Sum the digits
    soma = sum(last_digits)

    # Calcutate the final result
    result = soma % 4

    return str(result)


# Example Usage
if __name__ == "__main__":
    matriculas = ["2015310060", "2115080033", "2115080052", "2115080024"]

    result = identifier(matriculas)

    print("Result:", result)
