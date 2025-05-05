import eventlet
eventlet.monkey_patch()

import os
import base64
import cv2
import numpy as np
from collections import deque, defaultdict
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room
from emotion.detect import detect_emotion_from_image, label_encoder

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key')
socketio = SocketIO(app, cors_allowed_origins='*')

# For smoothing predictions
emotion_buffer = defaultdict(lambda: deque(maxlen=5))

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('join_room')
def handle_join(data):
    user = data.get('user', 'Anon')
    room = data.get('room')
    join_room(room)
    emit('system_message', {'text': f'{user} joined room {room}'}, room=room)

@socketio.on('emotion_ping')
def handle_emotion(data):
    image_data = data.get('image')
    user = data.get('user', 'Anon')
    room = data.get('room')

    try:
        _, encoded = image_data.split(',', 1)
        img = base64.b64decode(encoded)
        arr = np.frombuffer(img, np.uint8)
        if arr.size == 0:
            raise ValueError("Empty image buffer")
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        emit('emotion_update', {'emotion': 'No Face'}, room=request.sid)
        return

    emotion, face_crop, scores = detect_emotion_from_image(frame)

    if scores is not None:
        sid = request.sid
        emotion_buffer[sid].append(scores)
        avg_scores = np.mean(emotion_buffer[sid], axis=0)
        smoothed = label_encoder.inverse_transform([np.argmax(avg_scores)])[0]
        emit('emotion_update', {'emotion': smoothed}, room=request.sid)
    else:
        emit('emotion_update', {'emotion': emotion}, room=request.sid)

@socketio.on('send_message')
def handle_message(data):
    user = data.get('user', 'Anon')
    room = data.get('room')
    text = data.get('text', '')
    image_data = data.get('image', '')
    frame = None
    if image_data:
        try:
            _, encoded = image_data.split(',', 1)
            img = base64.b64decode(encoded)
            arr = np.frombuffer(img, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            pass

    emotion, face_crop, _ = detect_emotion_from_image(frame) if frame is not None else ('No Face', None, None)
    payload = {'user': user, 'text': text, 'emotion': emotion, 'image': None}
    if face_crop is not None:
        _, buffer = cv2.imencode('.jpg', face_crop)
        payload['image'] = base64.b64encode(buffer).decode('utf-8')

    emit('receive_message', payload, room=room)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
