import numpy as np


class Perceptron:
    def __init__(
        self, data: np.ndarray, learning_rate: float = 0.1, activation_function=None
    ):
        self.learning_rate = learning_rate
        self.activation_function = (
            activation_function if activation_function else self.step_function
        )

        self._data = None
        self.data = data  # This will call the setter and initialize _data and weights

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("Data must be a numpy array")

        if value.ndim != 1 or not all(
            isinstance(i, tuple) and len(i) == 2 for i in value
        ):
            raise ValueError(
                "Data must be a 1-dimensional numpy array of tuples (input, output)"
            )

        self._data = value
        # Reinitialize weights based on the input length of the first tuple
        self.weights = np.zeros(len(value[0][0]))

    @property
    def input_data(self) -> np.ndarray:
        return np.array((item[0] for item in self._data))

    @property
    def output_data(self) -> np.ndarray:
        return np.array((item[1] for item in self._data))

    def train(self, max_epochs: int = None):
        raise NotImplementedError("The train method is not implemented yet.")

    def predict(self, data: np.ndarray):
        raise NotImplementedError("The predict method is not implemented yet.")

    def run_single_epoch(self):
        raise NotImplementedError("The run_single_epoch method is not implemented yet.")

    @staticmethod
    def step_function(x: float) -> int:
        return 1 if x >= 0 else 0
