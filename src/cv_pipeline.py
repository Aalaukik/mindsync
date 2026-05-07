import cv2
import numpy as np
import tensorflow as tf

class EmotionClassifier:
    def __init__(self, model_path="models/mobilenet_emotion.h5"):
        # Load your lightweight model
        self.model = tf.keras.models.load_model(model_path)
        self.classes = ['Neutral', 'Focused', 'Confused', 'Distracted', 'Frustrated']
        # OpenCV face detector for cropping
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def predict_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None

        # Process the first detected face
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (224, 224)) # MobileNetV2 input size
        face_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(face_resized, axis=0))
        
        predictions = self.model.predict(face_array, verbose=0)
        state_idx = np.argmax(predictions[0])
        return self.classes[state_idx]