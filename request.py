import requests

# URL API
url = 'http://127.0.0.1:5000/recognize'

# Mengirim gambar
with open("uzi.png", "rb") as image_file:
    response = requests.post(url, files={"image": image_file})

# Hasil prediksi
print(response.json())
