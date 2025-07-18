import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import json

# Inisialisasi hand landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Folder dataset
DATASET_DIR = r"C:\Users\Aldan\Desktop\ASLClassification\asl_dataset"
OUTPUT_JSON = "output_hand_landmarks.json"

# Simpan semua hasil
results = []

# Iterasi setiap folder kelas
for label in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label)
    print(label_path)
    if not os.path.isdir(label_path):
        continue

    # Iterasi setiap gambar dalam folder kelas
    for image_file in os.listdir(label_path):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(label_path, image_file)

        # Baca gambar
        try:
            image = mp.Image.create_from_file(image_path)
            detection_result = detector.detect(image)

            # Ambil landmark (pakai tangan pertama saja jika ada)
            if detection_result.hand_landmarks:
                for hand in detection_result.hand_landmarks:
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand]
                    print("label: "+label)
                    results.append({
                        "image_path": os.path.join(label, image_file),
                        "landmarks": landmarks,
                        "label": label
                    })
        except Exception as e:
            print(f"Gagal memproses {image_path}: {e}")

# Simpan ke file JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"Ekstraksi selesai. Hasil disimpan di {OUTPUT_JSON}")