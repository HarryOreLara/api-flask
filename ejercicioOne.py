import numpy as np

# Datos de entrenamiento
X = np.array([[1, 2], [2, 3], [3, 1], [4, 3]])
y = np.array([0, 0, 1, 1])

# Inicialización de pesos y sesgo
np.random.seed(0)
pesos = np.random.rand(2)
sesgo = np.random.rand(1)

# Hiperparámetros
tasa_aprendizaje = 0.01
epochs = 1000

# Entrenamiento
for epoch in range(epochs):
    for i in range(len(X)):
        entrada = X[i]
        salida_real = y[i]
        
        # Propagación hacia adelante
        z = np.dot(entrada, pesos) + sesgo
        salida_predicha = 1 if z > 0 else 0
        
        # Cálculo del error
        error = salida_real - salida_predicha
        
        # Actualización de pesos y sesgo utilizando el descenso de gradiente
        pesos += tasa_aprendizaje * error * entrada
        sesgo += tasa_aprendizaje * error

# Prueba de la neurona entrenada
entrada_nueva = np.array([2, 2])
z_nuevo = np.dot(entrada_nueva, pesos) + sesgo
salida_nueva = 1 if z_nuevo > 0 else 0

print("Salida predicha para la entrada nueva:", salida_nueva)
