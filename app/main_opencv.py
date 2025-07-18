import cv2
import mediapipe as mp
import time
from hand_detector import HandDetector
from predict_class import prediksi_klasifikasi

# Inisialisasi hand detector
detector = HandDetector(r"C:\Users\Aldan\Desktop\ASLClassification\Code\hand_landmarker.task")

# Fungsi untuk menggambar landmark
def draw_landmarks(frame, landmarks, width, height):
    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        
def ubah_ke_fitur_rel_satu(landmarks):
    import numpy as np
    data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    anchor = data[0]
    rel = data[1:] - anchor
    return rel.flatten()


# Jalankan webcam
cap = cv2.VideoCapture(1) 
print("Tekan [Esc] untuk keluar...")
clasify = [
"0",
"1",
"2",
"3",
"4",
"5",
"6",
"7",
"8",
"9",
"a",
"b",
"c",
"d",
"e",
"f",
"g",
"h",
"i",
"j",
"k",
"l",
"m",
"n",
"o",
"p",
"q",
"r",
"s",
"t",
"u",
"v",
"w",
"x",
"y",
"z",
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah frame ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Deteksi
    timestamp_ms = int(time.time() * 1000)
    detector.detect(mp_image, timestamp_ms)

    # Ambil hasil dan gambar
    landmarks = detector.get_landmarks()

    if landmarks:
        draw_landmarks(frame, landmarks, frame.shape[1], frame.shape[0])

        # Ambil koordinat x, y, z → bentuk 1D list (63 elemen)
        data_mentah = []
        for lm in landmarks:
            data_mentah.extend([lm.x, lm.y, lm.z])

        # Prediksi kelas
        fitur = ubah_ke_fitur_rel_satu(landmarks)
        hasil_prediksi = prediksi_klasifikasi(fitur)
        print("👉 Kelas prediksi:", hasil_prediksi)


        # (opsional) Tampilkan kelas di layar
        cv2.putText(frame, f"Kelas: {clasify[hasil_prediksi]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
    # Tampilkan ke layar
    cv2.imshow("Hand Landmark Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()