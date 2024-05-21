from perceptron import Perceptron
import pandas as pd
import numpy as np


data = np.fromfile('./dataAll.txt')
data_reshaped = data.reshape(1000, 3)
data_reshaped

formated_data = ([ data_reshaped[:, 0], data_reshaped[:, 1]], data_reshaped[:, 2])

combined = list(zip(data_reshaped[:, 0], data_reshaped[:, 1]))

combined_all = list(zip(combined, data_reshaped[:, 2]))

combined_all = np.array(combined_all, dtype=object)

model = Perceptron(data=combined_all)

model.train()