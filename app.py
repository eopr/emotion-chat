import eventlet
eventlet.monkey_patch()

import os
import base64
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room
from emotion.detect import detect_emotion_from_image, label_encoder

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

@app.route('/')
def index():
    return render_template('index.html', theme='sleek-gray')

@socketio.on('join_room')
def handle_join(data):
    user = data.get('user', 'Anon')
    room = data.get('room')
    print(f"\U0001F465 {user} is joining room: {room}")
    join_room(room)
    emit('system_message', {
        'text': f'{user} joined room {room}',
        'time': datetime.now().strftime('%H:%M:%S')
    }, room=room)

@socketio.on('send_message')
def handle_message(data):
    user = data.get('user', 'Anon')
    room = data.get('room')
    text = str(data.get('text', '')).strip().replace('<', '').replace('>', '')
    image_data = str(data.get('image', '')).strip()

    if not text:
        return

    emotion = 'No Face'
    image_preview = None

    try:
        if image_data and len(image_data) > 100:
            _, encoded = image_data.split(',', 1)
            img = base64.b64decode(encoded)
            arr = np.frombuffer(img, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                emotions = []
                for _ in range(5):
                    detected_emotion, face_crop, _ = detect_emotion_from_image(frame)
                    emotions.append(detected_emotion)
                # Pick most frequent emotion
                if emotions:
                    emotion = max(set(emotions), key=emotions.count)
                if face_crop is not None:
                    _, buffer = cv2.imencode('.jpg', face_crop)
                    image_preview = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print("Emotion detection failed:", str(e))

    timestamp = datetime.now().strftime('%H:%M:%S')
    payload = {
        'user': user,
        'text': text,
        'emotion': emotion,
        'image': image_preview,
        'time': timestamp,
        'style': f'user-{user.lower().replace(" ", "-")}'
    }
    emit('receive_message', payload, room=room)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
