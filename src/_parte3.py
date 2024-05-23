import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

from perceptron import Perceptron
from utils import read_data, plot_results, train_test_split


def task_3():
    data = read_data("Holdout")

    print("Quantidade de pontos lidos: ", len(data))
    print("Formato dos dados: ", data.shape)
    print("\nExemplos de dados: ", *[(x, y) for x, y in data[:5]], sep="\n", end="\n\n")

    auxiliary = Perceptron(data)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_results(auxiliary.input_data, auxiliary.output_data, ax=ax)
    ax.set_title("Problema não linearmente separável")

    plt.tight_layout()
    plt.show()

    train_data, test_data = train_test_split(data)
    print("Quantidade de pontos para treinamento: ", len(train_data))
    print("Quantidade de pontos para teste: ", len(test_data))

    perceptron = Perceptron(train_data)
    perceptron.randomize_weights(floor=-0.5, ceiling=0.501)
    print("Vetor de pesos após a randomização: ", perceptron.weights)

    epoch, updates = perceptron.shuffle_train(100)

    print("\nTreinamento finalizado!")
    print("Quantidade de épocas treinadas: ", epoch)
    print("Quantidade de ajustes no vetor de pesos: ", updates)
    print("Quantidade de ajustes em cada época: ", perceptron.updates_per_epoch)

    auxiliary.data = test_data

    predictions = []
    for test in auxiliary._input_data:
        predictions.append(perceptron.predict(test))

    accuracy = accuracy_score(auxiliary._output_data, predictions)
    f1 = f1_score(auxiliary._output_data, predictions)
    recall = recall_score(auxiliary._output_data, predictions)
    precision = precision_score(auxiliary._output_data, predictions)

    print("Acurácia:", accuracy)
    print("F1-Score:", f1)
    print("Revocação:", recall)
    print("Precisão:", precision)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax_train, ax_test = axes

    plot_results(
        perceptron.input_data, perceptron.output_data, perceptron.weights, ax=ax_train
    )
    ax_train.set_title("Dados de treinamento")

    plot_results(
        auxiliary.input_data, auxiliary.output_data, perceptron.weights, ax=ax_test
    )
    ax_test.set_title("Dados de teste")

    plt.show()


if __name__ == "__main__":
    task_3()
