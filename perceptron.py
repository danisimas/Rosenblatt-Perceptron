import numpy as np


class Perceptron:
    def __init__(
        self,
        input_data: np.ndarray,
        output_value: np.ndarray,
        learning_rate: float = 0.1,
        activation_function=None,
    ):

        self._input_data = input_data
        self.output_value = output_value

        self.weights = np.zeros(input_data.shape[1])

        self.learning_rate = learning_rate

        self.activation_function = (
            activation_function if activation_function else self.step_function
        )

    @property
    def input_data(self) -> np.ndarray:
        return self._input_data

    @input_data.setter
    def input_data(self, value: np.ndarray):
        self._input_data = value
        self.weights = np.zeros(value.shape[1])

    def train(self, max_epochs: int = None):
        raise NotImplementedError("The train method is not implemented yet.")

    def predict(self, data: np.ndarray):
        raise NotImplementedError("The predict method is not implemented yet.")

    def run_single_epoch(self):
        raise NotImplementedError("The run_single_epoch method is not implemented yet.")

    @staticmethod
    def step_function(x: float) -> int:
        return 1 if x >= 0 else 0
