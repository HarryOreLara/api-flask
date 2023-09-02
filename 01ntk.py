import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

# Descargar datos adicionales necesarios (puede requerir conexión a internet)
nltk.download('punkt')
nltk.download('stopwords')

# Datos de entrenamiento
sentences = ["Esta película es genial.", "No me gustó la actuación de los actores.", "La música era increíble.", "Fue una experiencia terrible."]
labels = [1, 0, 1, 0]  # 1 para positivo, 0 para negativo

# Preprocesamiento de texto
stop_words = set(stopwords.words('spanish'))
X = []
for sentence in sentences:
    tokens = word_tokenize(sentence)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    X.append(tokens)

# Crear un vocabulario
word_set = set()
for tokens in X:
    word_set.update(tokens)

vocab_size = len(word_set)

# Crear una representación one-hot de las oraciones
X_encoded = []
for tokens in X:
    sentence_encoding = [1 if word in tokens else 0 for word in word_set]
    X_encoded.append(sentence_encoding)

# Convertir a matrices NumPy
X_encoded = np.array(X_encoded)
y = np.array(labels)

# Definir el modelo de la red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=vocab_size, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_encoded, y, epochs=100)

# Hacer predicciones
new_sentences = ["Me encantó la película.", "No puedo soportar esta película."]
new_X = []
for sentence in new_sentences:
    tokens = word_tokenize(sentence)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    new_X.append(tokens)

new_X_encoded = []
for tokens in new_X:
    sentence_encoding = [1 if word in tokens else 0 for word in word_set]
    new_X_encoded.append(sentence_encoding)

new_X_encoded = np.array(new_X_encoded)

predictions = model.predict(new_X_encoded)

for i, sentence in enumerate(new_sentences):
    sentiment = "positivo" if predictions[i] > 0.5 else "negativo"
    print(f"La oración '{sentence}' tiene un sentimiento {sentiment}.")
