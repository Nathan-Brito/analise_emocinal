import os
from flask import Flask, request, jsonify
from fer import FER
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

app = Flask(__name__)
detector = FER()

@app.route('/')
def index():

    return "Bem-vindo à API"

@app.route('/analise_emocional', methods=['POST'])
def analisar_emocao():
    if 'image' not in request.files:
        return jsonify({'error': ' '}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_array = np.array(img)

    emotion, score = detector.top_emotion(img_array)

    if emotion is None:
        return jsonify({'error': 'Nenhuma emoção detectada'}), 400
    return jsonify({
        'emocao': emotion,
        'porcentagem': f"{score * 100:.0f}%"
    })

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))