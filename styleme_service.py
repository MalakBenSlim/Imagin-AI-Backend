import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL_PATH = os.getenv('STYLEME_MODEL_PATH', 'saved_models/style_me_colab.keras')

app = Flask(__name__)

print("Chargement du modèle STYLE ME...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ STYLE ME — modèle chargé")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400

    file = request.files['image']
    img_pil = Image.open(file.stream).convert("RGB")

    ishape = model.input_shape
    th, tw = (ishape[1] or 224), (ishape[2] or 224)

    arr = np.array(img_pil.resize((tw, th)), dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, 0)

    preds = model.predict(batch, verbose=0)
    preds_list = preds[0].tolist()

    return jsonify({'preds': preds_list}), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None}), 200


if __name__ == '__main__':
    app.run(port=int(os.getenv('STYLEME_PORT', 5002)))
