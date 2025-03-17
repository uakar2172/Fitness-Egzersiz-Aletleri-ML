from flask import Flask, request, jsonify
#from flask_cors import CORS  # Flask-CORS'ı import ettik
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Flask uygulaması başlatma
app = Flask(__name__)

# CORS'u Flask uygulamasına ekleyin
#CORS(app)  # Tüm origin'lere izin verir

# Modeli yükleme
model = tf.keras.models.load_model('best_model.keras')

# Sınıf adlarını almak için train_generator'ı kullanın
class_names = ['Barbell', 'Cable Crossover Machine', 'Chest Fly Machine', 'Chest Press Machine',
               'Dikey Bisiklet', 'Dumbbell', 'Eliptik Bisiklet', 'Hand Gripper',
               'Koşu Bandı', 'Lat Pulldown Machine', 'Leg Curl Machine',
               'Leg Extension Machine', 'Leg Press Machine', 'Mat']

# Resmi ön işleme fonksiyonu
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resmi 224x224 boyutuna getir
    img_array = image.img_to_array(img)  # Resmi numpy dizisine dönüştür
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutunu ekle
    img_array /= 255.0  # Resmi normalize et
    return img_array


@app.after_request
def add_cors_headers(response):
    # CORS başlıklarını manuel olarak ekleyin
    response.headers['Access-Control-Allow-Origin'] = '*'  # Tüm kaynaklara izin verir
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response
# API'yi çalıştıracak route
@app.route('/predict', methods=['POST'])
def predict():
    # İstekten resmi al
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Resmi geçici olarak kaydet
    img_path = os.path.join('temp.jpg')
    file.save(img_path)

    # Resmi ön işleme
    img_array = prepare_image(img_path)

    # Modeli kullanarak tahmin yap
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Tahmin edilen sınıfın ismini al
    predicted_class_name = class_names[predicted_class_index]

    # Tahmin edilen sınıf ismini döndür
    return jsonify({'predicted_class': predicted_class_name})

# Test endpoint
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test endpoint çalışıyor!"})

if __name__ == '__main__':
    app.run(debug=True)
