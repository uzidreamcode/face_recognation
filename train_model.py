import os
import face_recognition
import pickle

# Folder dataset
DATASET_DIR = "dataset"

# List untuk menyimpan encoding wajah dan labelnya
known_face_encodings = []
known_face_names = []

# Loop untuk setiap orang di dalam dataset
for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    
    # Tambahkan pengecekan untuk memastikan ini adalah direktori
    if os.path.isdir(person_dir):  
        # Loop untuk setiap gambar di folder orang tersebut
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            
            # Load gambar dan encode wajah
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(person_name)

# Simpan encoding wajah dan nama orang ke dalam file
with open("face_encodings.pkl", "wb") as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Training selesai. Data wajah disimpan.")
