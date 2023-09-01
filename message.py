from flask import Flask, request, jsonify
from pymongo import MongoClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer





app = Flask(__name__)

# Descargar los recursos necesarios de NLTK
nltk.download('vader_lexicon')

# Inicializar el analizador de sentimientos de NLTK
sia = SentimentIntensityAnalyzer()


@app.route('/analizar_emocion', methods=['POST'])
def analizar_emocion():
    data = request.get_json()
    mensaje = data.get('mensaje')

    if mensaje:
        # Analizar la emoción del mensaje
        emocion = analizar_sentimiento(mensaje)
        return jsonify({'emocion': emocion}), 200
    else:
        return jsonify({'error': 'Falta el mensaje en la solicitud'}), 400

def analizar_sentimiento(texto):
    # Utilizar el analizador de sentimientos de NLTK
    sentiment = sia.polarity_scores(texto)
    
    # Determinar la emoción según el puntaje de polaridad
    if sentiment['compound'] >= 0.05:
        return 'positivo'
    elif sentiment['compound'] <= -0.05:
        return 'negativo'
    else:
        return 'neutral'

if __name__ == '__main__':
    app.run(debug=True)









# # Configuración de la conexión a MongoDB
# client = MongoClient("mongodb+srv://harry:root@cluster0.aniauvv.mongodb.net/dicta?retryWrites=true&w=majority")  # Reemplaza con tu URL de conexión
# db = client["dicta"]  # Reemplaza con el nombre de tu base de datos
# mensajes_collection = db["mensajes"]  # Reemplaza con el nombre de tu colección

# @app.route("/enviar_mensaje", methods=["POST"])
# def enviar_mensaje():
#     data = request.get_json()
#     mensaje = data.get("mensaje")

#     if mensaje:
#         # Guardar el mensaje en MongoDB
#         mensajes_collection.insert_one({"mensaje": mensaje})
#         return jsonify({"mensaje": "Mensaje enviado correctamente"}), 200
#     else:
#         return jsonify({"error": "Falta el mensaje en la solicitud"}), 400


# if __name__ == "__main__":
#     app.run(debug=True)
