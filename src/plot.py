import matplotlib.pyplot as plt
import numpy as np

def plot_results(x_data, y_data, weights):
    plt.figure(figsize=(8,8))

    # Criar gráfico de dispersão dos pontos de entrada usando o parametro "data"
    class_0 = x_data[y_data == 0]
    class_1 = x_data[y_data == 1]

    plt.scatter(class_0[:, 1], class_0[:, 2], color='red', marker='x', label='Classe 0')

    plt.scatter(class_1[:, 1], class_1[:, 2], color='blue', marker='o', label='Classe 1')

    
    # Criar gráfico de linha da reta x2 = -(w1/w2)x1 + (w0/w2) gerada pelo perceptron
    w0, w1, w2 = weights

    x1_values = np.linspace(min(x_data[:, 1]) - 1, max(x_data[:, 1]) + 1, 2)
    x2_values = -(w1 / w2) * x1_values + (w0 / w2)
    
    plt.plot(x1_values, x2_values, color='black', linestyle='-', linewidth=2, label='Linha de decisão')

    # Adicionar títulos e legendas
    plt.title('Linha decisão e pontos do problema')
    
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.legend()
    plt.grid(True)
    plt.show()