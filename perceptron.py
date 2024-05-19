import numpy as np


class Perceptron:
    def __init__(
        self,
        data: np.ndarray,
        activation_function=None,
        bias: float = -1,
        learning_rate: float = 0.1,
    ):
        self._input_data = None
        self._output_data = None
        self.data = data  # This will call the setter and initialize _input_data and _output_data

        self.activation_function = (
            activation_function if activation_function else self.step_function
        )

        self.bias = bias
        self.learning_rate = learning_rate

    @property
    def data(self):
        return np.hstack((self._input_data, self._output_data))

    @data.setter
    def data(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("Data must be a numpy array")

        if not all(len(i) == 2 for i in value):
            raise ValueError("Data must be a numpy array of tuples (input, output)")

        self._input_data = np.array([item[0] for item in value])
        self._output_data = np.array([item[1] for item in value])

        print("Input: ", self._input_data)
        print("Output: ", self._output_data)

        self.init_weights()

    def init_weights(self):
        # Initialize weights with an additional element for the bias
        self.weights = np.zeros(len(self._input_data[0]) + 1)

    def randomize_weights(self):
        # Randomize weights with values uniformly distributed between -1 and 1
        self.weights = np.random.uniform(-1, 1, len(self.weights))

    @property
    def input_data(self) -> np.ndarray:
        return self._input_data

    @property
    def output_data(self) -> np.ndarray:
        return self._output_data

    def train(self, max_epochs: int = None):
        raise NotImplementedError("The train method is not implemented yet.")

    def predict(self, data: np.ndarray):
        raise NotImplementedError("The predict method is not implemented yet.")

    def run_single_epoch(self):
        raise NotImplementedError("The run_single_epoch method is not implemented yet.")

    @staticmethod
    def step_function(x: float) -> int:
        return 1 if x >= 0 else 0


# Example usage
if __name__ == "__main__":
    # Example data: List of tuples (input, output)
    example_data = [(np.array([2, 2]), 1), (np.array([4, 4]), 0)]

    # Convert the list of tuples to a NumPy array
    example_data = np.array(example_data, dtype=object)

    # Initialize the Perceptron
    perceptron = Perceptron(data=example_data)

    # Randomize the weights
    perceptron.randomize_weights()

    # Check the weights
    print("Randomized weights:", perceptron.weights)
