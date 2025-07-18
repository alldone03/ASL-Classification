import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv

# Inisialisasi model
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

DATASET_DIR = r"C:\Users\Aldan\Desktop\ASLClassification\asl_dataset"
CSV_OUTPUT = "hand_features.csv"

with open(CSV_OUTPUT, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = [f"{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
    writer.writerow(header)

    for label in os.listdir(DATASET_DIR):
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder): continue

        for file in os.listdir(folder):
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')): continue

            path = os.path.join(folder, file)
            try:
                image = mp.Image.create_from_file(path)
                result = detector.detect(image)

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]  # tangan pertama
                    row = [coord for lm in hand for coord in (lm.x, lm.y, lm.z)]
                    row.append(label)
                    writer.writerow(row)
                    print(path,row)
            except Exception as e:
                print(f"Error at {path}: {e}")