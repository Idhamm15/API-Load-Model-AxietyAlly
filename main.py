from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import mediapipe as mp

app = Flask(__name__)

# Inisialisasi MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Memuat model
model = load_model('pose_classification_model.h5')

# Fungsi untuk mengekstrak landmark pose
def extract_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        return np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).flatten()
    return np.zeros(33*3)

# Fungsi untuk prediksi pose
def predict_pose(image):
    landmarks = extract_pose_landmarks(image)
    landmarks = landmarks.reshape(1, -1)
    prediction = model.predict(landmarks)
    class_id = np.argmax(prediction)
    pose_classes = ['TreePose', 'TrianglePose', 'StandingForwardBendPose']
    return pose_classes[class_id]

# Route untuk halaman utama
@app.route('/')
def hello():
    return "Hello, World!"

# Route untuk prediksi pose dari video feed
@app.route('/video_feed', methods=['POST'])
def video_feed():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Membaca gambar dari request
    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Prediksi pose
    predicted_pose = predict_pose(image)
    
    return jsonify({'pose': predicted_pose})

if __name__ == '__main__':
    # Menjalankan aplikasi Flask dengan Gunicorn
    from gunicorn.app.base import BaseApplication

    class FlaskApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key, value)

        def load(self):
            return self.application

    options = {
        'bind': '127.0.0.1:5000',
        'workers': 4,
    }

    FlaskApplication(app, options).run()
