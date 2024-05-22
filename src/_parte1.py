from src.activation_functions import step_function
from src.perceptron import Perceptron
from src.utils import read_data


def task_1():
    data = read_data("All")

    print("Quantidade de pontos lidos: ", len(data))
    print("Formato dos dados: ", data.shape)
    print("\nExemplos de dados: ", *[(x, y) for x, y in data[:5]], sep="\n")

    perceptron = Perceptron(
        data=data,
        activation_function=step_function,
        bias=-1,
        learning_rate=0.1,
    )

    print("Vetor de pesos inicial: ", perceptron.weights)

    perceptron.randomize_weights(floor=-0.5, ceiling=0.501)
    print("Vetor de pesos após a randomização: ", perceptron.weights)

    epoch, updates = perceptron.train()

    print("Treinamento finalizado!")
    print("Quantidade de épocas até a convergência: ", epoch)
    print("Quantidade de ajustes no vetor de pesos: ", updates)
    print("Quantidade de ajustes em cada época: ", perceptron.change_track)

    print("Vetor de pesos final: ", perceptron.weights)


if __name__ == "__main__":
    task_1()
