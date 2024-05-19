import numpy as np


class Perceptron:
    def __init__(
        self,
        input_data: np.ndarray,
        output_value: np.ndarray,
        learning_rate: float = 0.1,
        activation_function=None,
    ):

        self.input_data = input_data
        self.output_value = output_value

        self.learning_rate = learning_rate

        self.activation_function = (
            activation_function if activation_function else self.step_function
        )

    @staticmethod
    def step_function(x: float) -> int:
        return 1 if x >= 0 else 0

    def train(self, max_epochs: int = None):
        raise NotImplementedError("The train method is not implemented yet.")

    def predict(self, data: np.ndarray):
        raise NotImplementedError("The predict method is not implemented yet.")

    def run_single_epoch(self):
        raise NotImplementedError("The run_single_epoch method is not implemented yet.")
