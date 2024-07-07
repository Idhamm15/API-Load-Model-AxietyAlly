from flask import Flask, request, jsonify, Response
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
    app.run(debug=True)
