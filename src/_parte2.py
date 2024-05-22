from utils import read_data,identifier, matriculas

# Carrega os dados a partir do calculo sugerido pelo nome das matriculas
file_index = identifier(matriculas)
data = read_data(file_index)