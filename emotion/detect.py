import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load pre-trained and optimized model
model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
emotion_model = load_model(model_path, compile=False)

# Labels
label_encoder = LabelEncoder()
label_encoder.fit(['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'])

# Load face detector once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion_from_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return 'No Face', None, None

    # Use largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face = gray[y:y + h, x:x + w]
    resized = cv2.resize(face, (48, 48)).astype('float32') / 255.0
    resized = np.expand_dims(resized, axis=(0, -1))

    try:
        pred = emotion_model.predict(resized, verbose=0)
        emotion = label_encoder.inverse_transform([np.argmax(pred)])[0]
        return emotion, face, pred[0]
    except Exception as e:
        print("Prediction error:", e)
        return 'Error', None, None
