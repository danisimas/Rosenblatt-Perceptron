import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

import itertools

from perceptron import Perceptron
from utils import read_data, identifier, plot_results


def train_with_parameters(perceptron: Perceptron, rates, ranges):
    results = dict()

    for range_bounds, learning_rate in itertools.product(ranges, rates):
        perceptron.learning_rate = learning_rate

        results[(learning_rate, range_bounds)] = []

        for _ in range(10):
            perceptron.randomize_weights(-range_bounds, range_bounds + 0.1)

            # Train the perceptron
            epochs, updates = perceptron.train()

            # Save training results
            results[(learning_rate, range_bounds)].append(
                {
                    "epoch_count": epochs,
                    "update_count": updates,
                    "final_weights": perceptron.weights,
                }
            )

    return results


def task_2():
    file_index = identifier(["2015310060", "2115080033", "2115080052", "2115080024"])
    data = read_data(file_index)

    perceptron = Perceptron(data)
    rates = [0.4, 0.1, 0.01]
    bounds = [100, 0.5]

    results = train_with_parameters(perceptron, rates, bounds)

    table = PrettyTable()
    table.field_names = [
        "Taxa de Aprendizado",
        "Intervalo de Pesos",
        "Quantidade de Ajustes",
        "Menor número de épocas para convergência",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, result in zip(axes.flat, results.items()):
        params, data = result
        learning_rate, bounds = params

        last_weight = data[-1]["final_weights"]

        mean_updates = np.mean([result["update_count"] for result in data])
        std_updates = np.std([result["update_count"] for result in data])
        min_epochs = np.min([result["epoch_count"] for result in data])

        table.add_row(
            [
                learning_rate,
                f"({-bounds:.1f}, {bounds:.1f})",
                f"{mean_updates:.1f} ± {std_updates:.1f}",
                min_epochs,
            ]
        )

        plot_results(perceptron.input_data, perceptron.output_data, last_weight, ax)
        ax.set_title(f"{learning_rate} × ({-bounds:.1f}, {bounds:.1f})")

    fig.suptitle("Comparação de resultados: η × I ")
    plt.tight_layout()
    plt.show()

    print(table)


if __name__ == "__main__":
    task_2()
