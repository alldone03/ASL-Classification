🧠 ASL-Classification
Klasifikasi American Sign Language (ASL) menggunakan model pembelajaran mesin untuk mengenali abjad dari A–Z berdasarkan gesture tangan. Proyek ini cocok untuk edukasi, pengembangan aplikasi penerjemah isyarat, atau sebagai dasar proyek AI di bidang inklusivitas.

🚀 Fitur
Deteksi abjad ASL dari gambar atau kamera real-time

Preprocessing gambar otomatis

Pelatihan model kustom

Akurasi evaluasi & confusion matrix

Prediksi real-time dari webcam (opsional)

📁 Struktur Folder
bash
Copy
Edit
ASL-Classification/
│
├── dataset/              # Dataset gambar tangan ASL (train/test)
├── models/               # Model terlatih (.h5, .pt, dll)
├── notebooks/            # Notebook Jupyter untuk pelatihan & evaluasi
├── scripts/              # Script Python untuk training, testing, inferensi
├── utils/                # Preprocessing, visualisasi, dll
├── requirements.txt      # Library yang dibutuhkan
├── README.md             # Dokumentasi proyek
└── app.py                # (Opsional) aplikasi GUI / streamlit
📦 Instalasi
bash
Copy
Edit
git clone https://github.com/namakamu/ASL-Classification.git
cd ASL-Classification
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
python scripts/predict.py --image test_img.jpg
4. (Opsional) Real-time Webcam
bash
Copy
Edit
python scripts/webcam_predict.py
🧪 Dataset
Dataset yang digunakan: American Sign Language Letters Dataset

css
Copy
Edit
dataset/
  ├── train/
  │    ├── A/
  │    ├── B/
  │    └── ...
  └── test/
       ├── A/
       └── ...
📊 Hasil
Akurasi: 97.5% pada data validasi

Model terbaik: CNN 3-layer, trained 25 epochs

Contoh confusion matrix tersedia di notebooks/

✅ To-Do
 Training & evaluasi dasar

 GUI interaktif dengan Streamlit

 Integrasi webcam dengan OpenCV

 Export ke TensorFlow Lite / ONNX

📚 Referensi
Kaggle: ASL Alphabet

TensorFlow CNN Example

👨‍💻 Kontribusi
Pull Request dan issue sangat disambut!
Silakan fork, kembangkan, dan kirim PR jika ada ide perbaikan.

📜 Lisensi
MIT License © 2025 Aldan Prayogi
