import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import threading

latest_landmarks = None  # untuk simpan koordinat tangan terbaru


# === Inisialisasi model KNN ===
print('# ======== Load KNN Model ===========')
df = pd.read_csv("hand_features.csv")
X = df.drop('label', axis=1)
y = df['label']
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# === Variabel Global ===
current_prediction = None
lock = threading.Lock()  # untuk menghindari race condition

# === Callback untuk hasil deteksi tangan ===
def hand_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global current_prediction, latest_landmarks

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        features = [[coord for lm in hand for coord in (lm.x, lm.y, lm.z)]]

        # Simpan hasil prediksi
        with lock:
            current_prediction = model.predict(pd.DataFrame(features, columns=X.columns))[0]
            latest_landmarks = hand
    else:
        with lock:
            current_prediction = None
            latest_landmarks = None


# === Inisialisasi MediaPipe HandLandmarker ===
print('# ======== Inisialisasi MediaPipe HandLandmarker ===========')
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=hand_callback,
    num_hands=1,
)

detector = HandLandmarker.create_from_options(options)

def draw_landmarks(frame, landmarks, width, height):
    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # titik warna kuning

# === Mulai webcam ===
print('# ======== Mulai Webcam ===========')
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah ke RGB dan bungkus ke dalam mp.Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Kirim frame ke MediaPipe untuk deteksi async
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    detector.detect_async(mp_image, timestamp_ms)

    # Tampilkan prediksi
    with lock:
        if latest_landmarks:
            draw_landmarks(frame, latest_landmarks, frame.shape[1], frame.shape[0])

        if current_prediction:
            cv2.putText(frame, f"Gesture: {current_prediction}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
