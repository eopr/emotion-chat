import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.preprocessing import LabelEncoder

model_path = os.path.join(os.path.dirname(__file__), 'model.h5')

label_encoder = LabelEncoder()
label_encoder.fit(['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'])

def build_emotion_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model

emotion_model = build_emotion_model()
emotion_model.load_weights(model_path)
print("✅ model.h5 weights loaded successfully.")

def detect_emotion_from_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return 'No Face', None, None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face = gray[y:y + h, x:x + w]
    resized = cv2.resize(face, (48, 48)).reshape(1, 48, 48, 1) / 255.0
    pred = emotion_model.predict(resized)
    emotion = label_encoder.inverse_transform([np.argmax(pred)])[0]
    return emotion, face, pred[0]
