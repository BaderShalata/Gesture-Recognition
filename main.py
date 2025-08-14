from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model once
model = load_model('/Users/badershalata/Documents/MScDissertation/signlanguagemodel_256.keras')

actions = np.array(['hello', 'thanks', 'iloveyou', 'go', 'why', 'when', 'eat', 'today', 'help',
                    'friend', 'goodbye', 'understand', "don't-understand",
                    "i'm-good", 'how-are-you', 'my-name-is', 'good-morning',
                    'see-you-later'])
mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def sample_to_30_frames(sequence):
    indices = np.linspace(0, len(sequence) - 1, 30).astype(int)
    return sequence[indices]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'input' not in data:
        return jsonify({'error': 'No input video provided'}), 400

    base64_video = data['input']
    if ',' in base64_video:
        base64_video = base64_video.split(',')[1]

    try:
        video_bytes = base64.b64decode(base64_video)
    except Exception as e:
        return jsonify({'error': 'Invalid base64 video data'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)

    if not cap.isOpened():
        os.remove(temp_video_path)
        return jsonify({'error': 'Failed to open video file'}), 400

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        keypoints_list = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            keypoints = extract_keypoints(results)
            keypoints_list.append(keypoints)

            frame_count += 1

    cap.release()
    os.remove(temp_video_path)

    if len(keypoints_list) == 0:
        return jsonify({'error': 'No frames found in video or keypoints could not be extracted'}), 400

    all_keypoints = np.array(keypoints_list)

    if len(all_keypoints) >= 30:
        sampled_keypoints = sample_to_30_frames(all_keypoints)
    else:
        pad_len = 30 - len(all_keypoints)
        pad = np.zeros((pad_len, all_keypoints.shape[1]))
        sampled_keypoints = np.vstack((all_keypoints, pad))

    input_data = np.expand_dims(sampled_keypoints, axis=0)

    prediction = model.predict(input_data)[0]

    # Top 3 predictions
    top_indices = prediction.argsort()[-3:][::-1]
    top_predictions = [(actions[i], float(prediction[i])) for i in top_indices]

    result = {
        'predicted_class': top_predictions[0][0],
        'confidence': top_predictions[0][1],
        'frame_count': frame_count,
        'top_3': [{'label': label, 'confidence': conf} for label, conf in top_predictions]
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
