import numpy as np
import tensorflow as tf
import joblib

def ubah_ke_fitur_rel_satu(landmarks):
    import numpy as np
    data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    anchor = data[0]
    rel = data[1:] - anchor  # relatif terhadap titik ke-0
    return rel.flatten()

# Load model dan scaler sekali saja (global)
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\Aldan\Desktop\ASLClassification\Code\model_klasifikasi.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load(r"C:\Users\Aldan\Desktop\ASLClassification\Code\scaler.pkl")

def prediksi_klasifikasi(data_mentah):
    """
    Menerima: data_mentah -> list (1 baris 63 fitur)
    Mengembalikan: label prediksi (int)
    """

    # Pastikan input bentuk (1, 63)
    data_np = np.array(data_mentah, dtype=np.float32).reshape(1, -1)
    data_scaled = scaler.transform(data_np)

    interpreter.set_tensor(input_details[0]['index'], data_scaled)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = int(np.argmax(output))
    return predicted_class