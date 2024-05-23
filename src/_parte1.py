import matplotlib.pyplot as plt

from perceptron import Perceptron
from utils import read_data, plot_results


def task_1():
    data = read_data("All")

    print("Quantidade de pontos lidos: ", len(data))
    print("Formato dos dados: ", data.shape)
    print("\nExemplos de dados: ", *[(x, y) for x, y in data[:5]], sep="\n", end="\n\n")

    perceptron = Perceptron(data)

    print("Vetor de pesos inicial: ", perceptron.weights)

    perceptron.randomize_weights(floor=-0.5, ceiling=0.501)
    print("Vetor de pesos após a randomização: ", perceptron.weights)

    epoch, updates = perceptron.train()

    print("\nTreinamento finalizado!")
    print("Quantidade de épocas até a convergência: ", epoch)
    print("Quantidade de ajustes no vetor de pesos: ", updates)
    print("Quantidade de ajustes em cada época: ", perceptron.updates_per_epoch)

    print("Vetor de pesos final: ", perceptron.weights)

    plot_results(perceptron.input_data, perceptron.output_data, perceptron.weights)
    plt.show()


if __name__ == "__main__":
    task_1()
