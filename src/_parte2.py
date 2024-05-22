import numpy as np
from prettytable import PrettyTable

from activation_functions import step_function
from perceptron import Perceptron
from plot import plot_results
from utils import read_data, identifier


def train_with_parameters(data, rates, ranges):
    results = dict()
    for learning_rate in rates:
        for range_bounds in ranges:
            epochs_list = []
            updates_list = []

            min_epochs = float("inf")
            for train in range(10):
                # Init the perceptron with the learning_rate
                perceptron = Perceptron(
                    data=data,
                    activation_function=step_function,
                    bias=-1,
                    learning_rate=learning_rate,
                )

                # Draft the weights
                perceptron.randomize_weights(
                    floor=-range_bounds, ceiling=range_bounds + 0.1
                )

                # Train the perceptron
                epoch, updates = perceptron.train()

                # Save epochs and updates
                epochs_list.append(epoch)
                updates_list.append(updates)

                # Track the minimum number of epochs
                if epoch < min_epochs:
                    min_epochs = epoch

                if train == 9:
                    plot_results(perceptron.input_data, perceptron.output_data, perceptron.weights)

            # Calculate mean and standard deviation
            mean_updates = np.mean(updates_list)
            std_updates = np.std(updates_list)

            # Save results in the dictionary
            results[(learning_rate, range_bounds)] = {
                "mean_updates": mean_updates,
                "std_updates": std_updates,
                "min_epochs": min_epochs,
            }

    return results


def task_2():
    file_index = identifier(["2015310060", "2115080033", "2115080052", "2115080024"])
    data = read_data(file_index)

    rates = [0.4, 0.1, 0.01]
    ranges = [100, 0.5]

    results = train_with_parameters(data, rates=rates, ranges=ranges)

    # Create a PrettyTable object
    table = PrettyTable()
    table.field_names = [
        "Learning Rate",
        "Weight Range",
        "Mean Updates",
        "Std Updates",
        "Min Epochs",
    ]

    # Add rows to the table
    for key, value in results.items():
        learning_rate, range_val = key
        table.add_row(
            [
                learning_rate,
                f"{-range_val} to {range_val}",
                value["mean_updates"],
                value["std_updates"],
                value["min_epochs"],
            ]
        )

    # Print the table
    print(table)


if __name__ == "__main__":
    task_2()
