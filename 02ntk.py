import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

# Descargar datos adicionales necesarios (puede requerir conexión a internet)
nltk.download('punkt')

# Datos de entrenamiento (oraciones de estado de ánimo y etiquetas)
oraciones_estado_animo = ["Me siento muy feliz hoy.", "Estoy un poco triste ahora.", "Estoy enojado por la situación."]
etiquetas = [1, 0, 0]  # 1 para buen ánimo, 0 para mal ánimo

# Preprocesamiento de texto
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return tokens

# Tokenización y preprocesamiento de las oraciones
X = []
for oracion in oraciones_estado_animo:
    tokens = preprocess_text(oracion)
    X.append(tokens)

# Crear un vocabulario
vocabulario = set(word for tokens in X for word in tokens)

# Crear una representación one-hot de las oraciones
X_encoded = []
for tokens in X:
    oracion_encoding = [1 if word in tokens else 0 for word in vocabulario]
    X_encoded.append(oracion_encoding)

# Convertir a matrices NumPy
X_encoded = np.array(X_encoded)
y = np.array(etiquetas)

# Definir el modelo de la red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=len(vocabulario), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_encoded, y, epochs=100)

# Ejemplo de uso del modelo
texto_ingresado = input("Ingresa una oración sobre tu estado de ánimo: ")
tokens_ingresados = preprocess_text(texto_ingresado)
oracion_encoding = [1 if word in tokens_ingresados else 0 for word in vocabulario]
input_data = np.array([oracion_encoding])
prediction = model.predict(input_data)
resultado = "buen ánimo" if prediction > 0.5 else "mal ánimo"
print(f"La oración '{texto_ingresado}' representa un estado de ánimo de {resultado}.")
