# 🧠 ASL-Classification

Klasifikasi American Sign Language (ASL) menggunakan deep learning untuk mengenali abjad A–Z berdasarkan gesture tangan.  
Proyek ini cocok untuk edukasi, pengembangan aplikasi penerjemah bahasa isyarat, atau riset machine learning.

---

## 🚀 Fitur

- Deteksi abjad ASL dari gambar statis
- Model CNN sederhana dengan akurasi tinggi
- Evaluasi model menggunakan confusion matrix
- (Opsional) Prediksi real-time menggunakan webcam

---

## 📁 Struktur Folder

ASL-Classification/
│
├── dataset/ # Dataset gambar tangan ASL
├── models/ # Model terlatih (.h5, .pt, dll)
├── notebooks/ # Notebook Jupyter untuk training & evaluasi
├── scripts/ # Script Python: train, evaluate, predict
├── utils/ # Fungsi preprocessing, visualisasi, dll
├── app.py # (Opsional) aplikasi GUI / Streamlit
├── requirements.txt # Library yang dibutuhkan
└── README.md # Dokumentasi proyek

yaml
Copy
Edit

---

## 📦 Instalasi

1. Clone repositori:

```bash
git clone https://github.com/namakamu/ASL-Classification.git
cd ASL-Classification
Install dependensi:

bash
Copy
Edit
pip install -r requirements.txt
🔧 Cara Penggunaan
1. Training Model
bash
Copy
Edit
python scripts/train.py --epochs 25 --batch 32 --model cnn
2. Evaluasi Model
bash
Copy
Edit
python scripts/evaluate.py --model models/asl_cnn.h5
3. Prediksi dari Gambar
bash
Copy
Edit
python scripts/predict.py --image path_ke_gambar.jpg
4. (Opsional) Prediksi Webcam
bash
Copy
Edit
python scripts/webcam_predict.py
🧪 Dataset
Dataset: ASL Alphabet Dataset - Kaggle

Struktur dataset:

css
Copy
Edit
dataset/
  ├── train/
  │   ├── A/
  │   ├── B/
  │   └── ...
  └── test/
      ├── A/
      └── ...
📊 Hasil Model
Akurasi validasi: 97.5%

Model: CNN 3-layer

Epoch: 25

Visualisasi: confusion matrix tersedia di notebooks/

✅ Rencana Pengembangan
 Training model klasifikasi abjad

 Evaluasi performa dengan confusion matrix

 Real-time prediksi dari webcam

 Export ke TensorFlow Lite atau ONNX

 GUI menggunakan Streamlit

📚 Referensi
ASL Dataset - Kaggle

TensorFlow CNN Image Classification

👨‍💻 Kontribusi
Silakan fork, buka issue, atau kirim Pull Request untuk pengembangan proyek ini lebih lanjut 🙌

📜 Lisensi
MIT License © 2025 Aldan Prayogi

yaml
Copy
Edit
