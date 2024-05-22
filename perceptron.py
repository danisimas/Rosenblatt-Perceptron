import numpy as np


class Perceptron:
    def __init__(
        self,
        data: np.ndarray,
        activation_function=None,
        bias: float = -1,
        learning_rate: float = 0.1,
    ):
        """
        Initialize the Perceptron.

        Parameters:
            data (np.ndarray): The input data as a numpy array of tuples (input, output).
            activation_function (function, optional): The activation function. Defaults to None.
            bias (float, optional): The bias value. Defaults to -1.
            learning_rate (float, optional): The learning rate. Defaults to 0.1.
        """
        self.bias = bias
        self.learning_rate = learning_rate

        self.activation_function = (
            activation_function if activation_function else self.step_function
        )

        self._input_data = None
        self._output_data = None
        self.data = data  # This will call the setter and initialize _input_data and _output_data

        self.change_track = (
            []
        )  # Keep the information about the number of adjusts done in each epoch

    @property
    def data(self):
        """
        Get the input and output data as a combined numpy array.

        Returns:
            np.ndarray: Combined input and output data.
        """
        return np.hstack((self._input_data, self._output_data))

    @data.setter
    def data(self, value: np.ndarray):
        """
        Set the input and output data.

        Parameters:
            value (np.ndarray): The input data as a numpy array of tuples (input, output).
        """
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
        """
        Initialize the weights array.
        """
        self.weights = np.zeros(len(self._input_data[0]))

    def randomize_weights(self, floor=-0.5, ceiling=0.5):
        """
        Set each value of the weight array to a random number between the given interval
        """
        self.weights = np.random.uniform(floor, ceiling, len(self.weights))

    @property
    def input_data(self) -> np.ndarray:
        """
        Get the input data.

        Returns:
            np.ndarray: Input data.
        """
        return self._input_data

    @property
    def output_data(self) -> np.ndarray:
        """
        Get the output data.

        Returns:
            np.ndarray: Output data.
        """
        return self._output_data

    def train(self, max_epochs: int = None):
        """
        Train the Perceptron. Stops on max_epochs or on convergence.

        Parameters:
            max_epochs (int, optional): Maximum number of epochs. Defaults to None.

        Returns:
            (epoch, weight_updates): Number of epochs trained and the amount of updates applied to the weights array done.
        """
        self.weight_updates = 0
        last_weight_update = 0

        # Train for at maximum "max_epoch" epochs
        if max_epochs and max_epochs > 0:
            for epoch in range(max_epochs):
                self.__run_single_epoch()

                # No change means no value was incorrectly predicted and no more training is necessary
                if last_weight_update == self.weight_updates:
                    return epoch + 1, self.weight_updates

                last_weight_update = self.weight_updates
            return max_epochs, self.weight_updates

        # Train until done OR user decides to quit on multiple of 500

        epoch = 0
        while True:
            self.__run_single_epoch()
            epoch += 1

            if epoch % 500 == 0:
                choice = input(f"Trained for {epoch} epochs, continue? (y/n)")

                if choice in "nN":
                    return epoch, self.weight_updates

            # No change means no value was incorrectly predicted and no more training is necessary
            if last_weight_update == self.weight_updates:
                return epoch, self.weight_updates

            last_weight_update = self.weight_updates

    def predict(self, values: np.ndarray):
        """
        Predict the output for the given input values.

        Parameters:
            values (np.ndarray): Input values.

        Returns:
            Predicted output.
        """
        u = np.dot(values, self.weights)
        return self.activation_function(u)

    def __run_single_epoch(self):
        """
        Perform a single epoch of training.
        """
        start_w = self.weight_updates
        for input_values, output_value in zip(self._input_data, self._output_data):

            y = self.predict(input_values)

            if y == output_value:
                continue

            error = output_value - y

            self.weights = self.weights + self.learning_rate * error * input_values
            self.weight_updates += 1
        self.change_track.append(self.weight_updates - start_w)

    @staticmethod
    def step_function(x: float) -> int:
        """
        Step activation function.

        Parameters:
            x (float): Input value.

        Returns:
            int: Output value (0 or 1).
        """
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

    fits, epochs = perceptron.train()

    print(f"Finished training in {epochs} epochs with {fits} fits!")
    print("Final weights:", perceptron.weights)

    print("Adjusts done in each epoch:",perceptron.change_track)
