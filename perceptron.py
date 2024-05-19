import numpy as np


class Perceptron:
    def __init__(
        self,
        data: np.ndarray,
        activation_function=None,
        bias: float = -1,
        learning_rate: float = 0.1,
    ):
        self.bias = bias
        self.learning_rate = learning_rate

        self.activation_function = (
            activation_function if activation_function else self.step_function
        )

        self._input_data = None
        self._output_data = None
        self.data = data  # This will call the setter and initialize _input_data and _output_data

    @property
    def data(self):
        return np.hstack((self._input_data, self._output_data))

    @data.setter
    def data(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("Data must be a numpy array")

        if not all(len(i) == 2 for i in value):
            raise ValueError("Data must be a numpy array of tuples (input, output)")

        self._input_data = np.array(
            [np.insert(item[0], 0, self.bias) for item in value]
        )
        self._output_data = np.array([item[1] for item in value])

        self.__init_weights()

    def __init_weights(self):
        # Initialize weights with an additional element for the bias
        self.weights = np.zeros(len(self._input_data[0]))

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
        self.w = 0
        last_w = 0

        # Train for at maximum "max_epoch" epochs
        if max_epochs and max_epochs > 0:
            for _ in range(max_epochs):
                self.__run_single_epoch()

                # No change means no value was incorrectly predicted and no more training is necessary
                if last_w == self.w:
                    return self.w

                last_w = self.w
            return self.w

        # Train until done OR user decides to quit on multiple of 500
        while True:
            if self.w > 0 and self.w % 500 == 0:
                choice = input(f"Trained for {self.w} epochs, continue? (y/n)")

                if choice == "n":
                    return

            self.__run_single_epoch()

            # No change means no value was incorrectly predicted and no more training is necessary
            if last_w == self.w:
                return self.w

            last_w = self.w

    def predict(self, values: np.ndarray):
        u = np.dot(values, self.weights)
        return self.activation_function(u)

    def __run_single_epoch(self):
        for input_values, output_value in zip(self._input_data, self._output_data):

            y = self.predict(input_values)

            if y == output_value:
                continue

            error = output_value - y

            self.weights = self.weights + self.learning_rate * error * input_values
            self.w += 1

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

    epochs = perceptron.train()

    print(f"Finished training in {epochs} epochs!")
    print("Final weights:", perceptron.weights)
