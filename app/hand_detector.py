import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

class HandDetector:
    def __init__(self, model_path="hand_landmarker.task"):
        self.latest_landmarks = None

        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.hand_callback,
            num_hands=1
        )
        self.detector = HandLandmarker.create_from_options(self.options)

    def hand_callback(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if result.hand_landmarks:
            self.latest_landmarks = result.hand_landmarks[0]
        else:
            self.latest_landmarks = None

    def detect(self, mp_image, timestamp_ms):
        self.detector.detect_async(mp_image, timestamp_ms)

    def get_landmarks(self):
        return self.latest_landmarks