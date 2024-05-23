import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


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


def plot_results(x_data, y_data, weights=None, ax: Axes = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Criar gráfico de dispersão dos pontos de entrada usando os parametros "data"
    class_0 = x_data[y_data == 0]
    class_1 = x_data[y_data == 1]

    ax.scatter(class_0[:, 1], class_0[:, 2], color="r", marker="x", label="Classe 0")
    ax.scatter(class_1[:, 1], class_1[:, 2], color="b", marker="o", label="Classe 1")

    # Criar gráfico de linha da reta x2 = -(w1/w2)x1 + (w0/w2) gerada pelo perceptron
    if weights is not None:
        w0, w1, w2 = weights

        x1_min = np.min(x_data[:, 1]) - 0.5
        x1_max = np.max(x_data[:, 1]) + 0.5
        x1_values = np.linspace(x1_min, x1_max, 2)

        x2_min = np.min(x_data[:, 2]) - 0.5
        x2_max = np.max(x_data[:, 2]) + 0.5
        ax.set_ylim(x2_min, x2_max)

        x2_values = -(w1 / w2) * x1_values + (w0 / w2)

        ax.plot(
            x1_values,
            x2_values,
            color="black",
            linestyle="-",
            linewidth=2,
            label="Linha de decisão",
        )

    # Adicionar títulos e legendas
    ax.set_title("Linha de Decisão e Pontos do Problema")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax.grid(True)


def train_test_split(data: np.array, train_portion=0.7):
    # Make a copy from data
    copy = data.copy()

    # Shuffle the data
    np.random.shuffle(copy)

    # Calculate an edge for splitting
    edge = int(data.shape[0] * train_portion)

    # Return train and test portions respectively
    return copy[:edge], copy[edge:]


# Example Usage
if __name__ == "__main__":
    matriculas = ["2015310060", "2115080033", "2115080052", "2115080024"]

    result = identifier(matriculas)

    print("Result:", result)
