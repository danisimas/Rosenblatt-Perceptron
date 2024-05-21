def identifier(values):
    # Pega o último dígito de cada matrícula
    last_digits = (int(value[-1]) for value in values)
    
    # Soma os últimos dígitos
    soma = sum(last_digits)
    
    # Calcula o módulo 4 da soma'
    result = soma % 4    
    
    return str(result)

# Exemplo de uso
if __name__ == '__main__':        
    matriculas = ['2015310060', '2115080033', '2115080052', '2115080004']
    resultado = identifier(matriculas)
    print("Resultado:", resultado)  # Output: Resultado: 3

