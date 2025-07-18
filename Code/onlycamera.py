import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# === Inisialisasi komponen MediaPipe HandLandmarker ===
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Variabel global untuk menyimpan hasil landmark
latest_landmarks = None

# Fungsi callback saat tangan terdeteksi
def hand_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_landmarks
    if result.hand_landmarks:
        latest_landmarks = result.hand_landmarks[0]
        print("\nKoordinat 21 Titik Tangan (x, y, z):")
        for i, lm in enumerate(latest_landmarks):
            print(f"  Titik {i:2d}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}")
    else:
        latest_landmarks = None

# Setup model
model_path = "hand_landmarker.task"  # Ganti jika path model berbeda
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=hand_callback,
    num_hands=1
)

# Buat detektor
detector = HandLandmarker.create_from_options(options)

# Fungsi menggambar landmark ke frame
def draw_landmarks(frame, landmarks, width, height):
    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

# === Mulai Webcam ===
cap = cv2.VideoCapture(1)
print("Tekan [Esc] untuk keluar...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah ke RGB untuk MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Kirim ke MediaPipe untuk deteksi async
    timestamp_ms = int(time.time() * 1000)
    detector.detect_async(mp_image, timestamp_ms)

    # Gambar titik tangan jika ada
    if latest_landmarks:
        draw_landmarks(frame, latest_landmarks, frame.shape[1], frame.shape[0])

    # Tampilkan hasil ke layar
    cv2.imshow("Hand Landmark Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()