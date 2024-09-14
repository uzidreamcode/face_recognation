from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import pickle
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Load face encodings yang sudah dilatih
with open("face_encodings.pkl", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

app = Flask(__name__)
CORS(app)

# Route untuk prediksi wajah
@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    image_base64 = data['image']
    
    # Decode base64 image
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    image = np.array(image)
    
    # Ekstraksi wajah dari gambar
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Simpan hasil prediksi
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Jika ada wajah yang cocok
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return jsonify({'names': face_names})

if __name__ == '__main__':
    app.run(debug=True)
