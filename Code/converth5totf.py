import tensorflow as tf

# 1. Load model HDF5
model = tf.keras.models.load_model(r"C:\Users\Aldan\Desktop\ASLClassification\Code\Model_ASL.h5")

# 2. Konversi ke TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Simpan ke file .tflite
with open("model_klasifikasi.tflite", "wb") as f:
    f.write(tflite_model)

print("Model berhasil dikonversi ke TFLite!")