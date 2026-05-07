import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class EmotionClassifier:
    def __init__(self, model_path="models/mobilenet_emotion.h5"):
        self.classes = ['Neutral', 'Focused', 'Confused', 'Distracted', 'Frustrated']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 1. Rebuild the exact architecture used in Colab
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(5, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.load_weights(model_path)

        # --- OPTIMIZATION VARIABLES ---
        self.frame_skip = 5  # Only process 1 out of every 5 frames
        self.frame_count = 0
        self.last_prediction = None

    def predict_frame(self, frame):
        self.frame_count += 1

        # LATENCY FIX: Frame Skipping
        # If it's not the 5th frame, instantly return the last known emotion
        if self.frame_count % self.frame_skip != 0 and self.last_prediction is not None:
            return self.last_prediction

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Faster detection parameters (scaleFactor=1.3, minNeighbors=5)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            self.last_prediction = None
            return None

        # Process the first detected face
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (224, 224)) 
        
        # ACCURACY FIX: Match the training data scaling EXACTLY (0 to 1)
        face_array = np.expand_dims(face_resized, axis=0)
        face_array = face_array.astype('float32') / 255.0
        
        predictions = self.model.predict(face_array, verbose=0)
        state_idx = np.argmax(predictions[0])
        
        self.last_prediction = self.classes[state_idx]
        return self.last_prediction