# Rosenblatt Perceptron - UEA/EST RNA 2024.1

## Description
Class project with the objective of implementing a training algorithm using Rosenblatt's Perceptron and Supervised Learning for binary classification, from a course on **Artificial Neural Networks** @ [UEA - Amazonas State University](https://www2.uea.edu.br/).

## Summary
- [Installation](#installation)
- [Repository structure](#repository-structure)
- [Authors](#authors)

## Installation

### Recommended - Conda / Miniconda Environment
Download and install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or alternatively [miniconda](https://docs.anaconda.com/free/miniconda/) and load the `environment.yml` file from the root of this git repo.

```sh
conda env create -f environment.yml -n my_env_name
```

Once the environment has been created, activate it and run the code with:

```sh
conda activate my_env_name

# Run and see results for part I
python ./src/_parte1.py

# Run and see results for part II
python ./src/_parte2.py

# Run and see results for part III
python ./src/_parte3.py
```

You can also open our [Jupyter](https://docs.jupyter.org/en/latest/start/index.html) Notebook:

```sh
jupyter notebook ./entrega.ipynb
```

Read more on https://docs.jupyter.org/en/latest/running.html

## Repository structure

- `/src/` - Directory containing all of python source code developed for this project.
    - `/src/activation_functions.py` - Python module containing a heaviside step function for use as the default activation function of the Perceptron.
    - `/src/perceptron.py` - Python module containing a implementation of a Rosenblatt's Perceptron artificial neuron model, written as a Python class.
    - `/src/_parteN.py` - Each of these files corresponds to a section of the tasks described in the project's PDF file.
    - `/src/utils.py` - Python module implementing auxiliary functions which are used by each of the tasks.
- `entrega.ipynb` - Jupyter Notebook showcasing the Perceptron's implementation and experimentation with 3 different problems.
- `/task/` - Directory containing all training data used in each task and a PDF with the project's full description.

## Authors
- [Daniele Simas](https://github.com/danisimas)
- [Felipe Amorim](https://github.com/FMABr)
- [José Manuel](https://github.com/Bloco314)
- [Miguel Angelo](https://github.com/Miguel-Angelo-wq)

## Credits
- Elloá B. Guedes - [github](https://github.com/elloa) and [website](http://www.elloaguedes.com/)
