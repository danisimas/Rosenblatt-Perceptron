import numpy as np
from activation_functions import step_function
from perceptron import Perceptron
from utils import read_data, identifier, matriculas
from prettytable import PrettyTable

file_index = identifier(matriculas)
data = read_data(file_index)

n = [0.4, 0.1, 0.01]
i = [100, 0.5]

resultados = {}


def train_with_set():
    for learning_rate in n:
        for range_val in i:
            epochs_list = []
            updates_list = []
            min_epochs = float('inf')
            for train in range(10):
                # Init the perceptron with the learning_rate
                perceptron = Perceptron(
                    data=data,
                    activation_function=step_function,
                    bias=-1,
                    learning_rate=learning_rate,
                )

                # Draft the weights
                perceptron.randomize_weights(floor=-range_val, ceiling=range_val + 0.1)

                # Train the perceptron
                epoch, updates = perceptron.train()
                
                # Save epochs and updates
                epochs_list.append(epoch)
                updates_list.append(updates)

                # Track the minimum number of epochs
                if epoch < min_epochs:
                    min_epochs = epoch

                if train == 9:
                    # ToDo: plot the last solution
                    pass
            
            # Calculate mean and standard deviation
            mean_updates = np.mean(updates_list)
            std_updates = np.std(updates_list)

            # Save results in the dictionary
            resultados[(learning_rate, range_val)] = {
                'mean_updates': mean_updates,
                'std_updates': std_updates,
                'min_epochs': min_epochs,
            }

# Execute the training process
train_with_set()

# Create a PrettyTable object
table = PrettyTable()
table.field_names = ["Learning Rate", "Weight Range", "Mean Updates", "Std Updates", "Min Epochs"]

# Add rows to the table
for key, value in resultados.items():
    learning_rate, range_val = key
    table.add_row([
        learning_rate,
        f"{-range_val} to {range_val}",
        value['mean_updates'],
        value['std_updates'],
        value['min_epochs']
    ])

# Print the table
print(table)
