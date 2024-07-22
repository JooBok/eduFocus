import time, joblib
import requests, base64, cv2, logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import mediapipe as mp
import numpy as np
from threading import Lock

from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation
from pymongo import MongoClient

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGGREGATOR_URL = "http://result-aggregator-service/aggregate"
SESSION_SERVICE_URL = "http://session-service"
################################## Mongo setting ##################################
# MONGO_URI = 'mongodb://mongodb-service:27017'
MONGO_URI = 'mongodb://root:root@mongodb:27017/saliency_db?authSource=admin'
MONGO_DB = 'saliency_db'
# MONGO_COLLECTION = 'contents2'
################################ mediaPipe setting ################################
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
################################## model setting ##################################
global model_x, model_y
model_x = None
model_y = None
model_lock = Lock()

def load_models():
    global model_x, model_y
    with model_lock:
        if model_x is None or model_y is None:
            model_x, model_y = joblib.load('/app/model/gaze_model.pkl')

load_models()
################################## Mongo setting ##################################
def mongodb_client():
    return MongoClient(MONGO_URI)

def extract_saliencyMap(video_id):
    client = mongodb_client()
    db = client[MONGO_DB]
    collection = db[video_id]
    
    all_data = collection.find()
    extracted_data = []

    for document in all_data:
        frame_num = document.get('frame_num')
        saliency_map = document.get('saliency_map')
        
        if frame_num is not None and saliency_map is not None:
            extracted_data.append([frame_num, saliency_map])
        else:
            logger.error("Error: 'frame_num' or 'saliency_map' not found in the document")

    return extracted_data
################################### Session ###################################
class GazeSession:
    def __init__(self):
        self.gaze_buffer = GazeBuffer()
        self.gaze_fixation = GazeFixation()
        self.gaze_sequence = []
        self.prev_gaze = None
        self.final_result = {}
        self.sequence_length = 10

    def to_dict(self):
        return {
            'gaze_buffer': self.gaze_buffer.to_dict(),
            'gaze_fixation': self.gaze_fixation.to_dict(),
            'gaze_sequence': [gaze.tolist() for gaze in self.gaze_sequence],
            'prev_gaze': self.prev_gaze.tolist() if self.prev_gaze is not None else None,
            'final_result': {str(k): v for k, v in self.final_result.items()},
            'sequence_length': self.sequence_length
        }

    @classmethod
    def from_dict(cls, data):
        session = cls()
        session.gaze_buffer = GazeBuffer.from_dict(data.get('gaze_buffer', {}))
        session.gaze_fixation = GazeFixation.from_dict(data.get('gaze_fixation', {}))
        session.gaze_sequence = [np.array(gaze) for gaze in data.get('gaze_sequence', [])]
        session.prev_gaze = np.array(data['prev_gaze']) if data.get('prev_gaze') is not None else None
        session.final_result = {int(k): v for k, v in data.get('final_result', {}).items()}
        session.sequence_length = data.get('sequence_length', 10)
        return session

def get_session(session_id):
    response = requests.get(f"{SESSION_SERVICE_URL}/get_session/{session_id}")
    if response.status_code == 200:
        session_data = response.json()
        gaze_data = session_data['components'].get('gaze_tracking', {})
        return GazeSession.from_dict(gaze_data)
    elif response.status_code == 404:
        logger.warning(f"Session not found. Creating new session for {session_id}")
        return GazeSession()
    else:
        logger.error(f"Failed to get session. Status code: {response.status_code}, Response: {response.text}")
        return GazeSession()

def update_session(session_id, gaze_session):
    update_data = {
        "component": "gaze_tracking",
        "data": gaze_session.to_dict()
    }
    response = requests.put(f"{SESSION_SERVICE_URL}/update_session/{session_id}", json=update_data)
    if response.status_code == 200:
        logger.info(f"Successfully updated session for {session_id}")
    elif response.status_code == 404:
        logger.warning(f"Session not found when updating. Creating new session for {session_id}")
        create_session(session_id, gaze_session)
    else:
        logger.error(f"Failed to update session. Status code: {response.status_code}, Response: {response.text}")

def create_session(session_id, gaze_session):
    create_data = {
        "session_id": session_id,
        "components": {
            "gaze_tracking": gaze_session.to_dict()
        }
    }
    response = requests.post(f"{SESSION_SERVICE_URL}/create_session", json=create_data)
    if response.status_code == 201:
        logger.info(f"Successfully created new session for {session_id}")
    else:
        logger.error(f"Failed to create session. Status code: {response.status_code}, Response: {response.text}")

################################# GBR 모델 인풋 만드는 class, 함수 #################################
class GazeBuffer:
    def __init__(self, buffer_size=3, smoothing_factor=0.3):
        self.buffer = []
        self.buffer_size = buffer_size
        self.smoothing_factor = smoothing_factor
        self.previous_gaze = None
    def add(self, gaze):
        self.buffer.append(gaze)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    def get_average(self):
        if not self.buffer:
            return None
        current_average = np.mean(self.buffer, axis=0)
        if self.previous_gaze is None:
            self.previous_gaze = current_average
        else:
            current_average = (1 - self.smoothing_factor) * current_average + self.smoothing_factor * self.previous_gaze
            self.previous_gaze = current_average
        return current_average
    def to_dict(self):
        return {
            'buffer': [gaze.tolist() for gaze in self.buffer],
            'buffer_size': self.buffer_size,
            'smoothing_factor': self.smoothing_factor,
            'previous_gaze': self.previous_gaze.tolist() if self.previous_gaze is not None else None
        }
    @classmethod
    def from_dict(cls, data):
        buffer = cls(data.get('buffer_size', 3), data.get('smoothing_factor', 0.3))
        buffer.buffer = [np.array(gaze) for gaze in data.get('buffer', [])]
        buffer.previous_gaze = np.array(data['previous_gaze']) if data.get('previous_gaze') is not None else None
        return buffer

class GazeFixation:
    def __init__(self, velocity_threshold=0.1, duration=0.2, window_size=6):
        self.velocity_threshold = velocity_threshold
        self.duration = duration
        self.window_size = window_size
        self.gaze_history = []
        self.time_history = []
        self.start_time = None
    def update(self, gaze):
        current_time = time.time()
        self.gaze_history.append(gaze)
        self.time_history.append(current_time)
        if len(self.gaze_history) > self.window_size:
            self.gaze_history.pop(0)
            self.time_history.pop(0)

        if len(self.gaze_history) < self.window_size:
            return False
        velocities = []
        for i in range(1, len(self.gaze_history)):
            dist = np.linalg.norm(np.array(self.gaze_history[i]) - np.array(self.gaze_history[i-1]))
            time_diff = self.time_history[i] - self.time_history[i-1]
            velocities.append(dist / time_diff if time_diff > 0 else 0)     
        avg_velocity = np.mean(velocities)
        if avg_velocity < self.velocity_threshold:
            if self.start_time is None:
                self.start_time = current_time
            elif current_time - self.start_time >= self.duration:
                return True
        else:
            self.start_time = None
        return False
    def to_dict(self):
        return {
            'velocity_threshold': self.velocity_threshold,
            'duration': self.duration,
            'window_size': self.window_size,
            'gaze_history': [gaze.tolist() for gaze in self.gaze_history],
            'time_history': self.time_history,
            'start_time': self.start_time
        }
    @classmethod
    def from_dict(cls, data):
        fixation = cls(
            data.get('velocity_threshold', 0.1),
            data.get('duration', 0.2),
            data.get('window_size', 6)
        )
        fixation.gaze_history = [np.array(gaze) for gaze in data.get('gaze_history', [])]
        fixation.time_history = data.get('time_history', [])
        fixation.start_time = data.get('start_time')
        return fixation

def calculate_distance(iris_landmarks, image_height):
    left_iris, right_iris = iris_landmarks
    distance = np.linalg.norm(np.array(left_iris) - np.array(right_iris))
    estimated_distance = (1 / distance) * image_height
    return estimated_distance

def get_center(landmarks):
    return np.mean([[lm.x, lm.y, lm.z] for lm in landmarks], axis=0)

def estimate_gaze(eye_center, iris_center, estimated_distance):
    eye_vector = iris_center - eye_center
    gaze_point = eye_center + eye_vector * estimated_distance
    return gaze_point

def estimate_head_pose(face_landmarks):
    nose = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y, face_landmarks.landmark[1].z])
    left_eye = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y, face_landmarks.landmark[33].z])
    right_eye = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y, face_landmarks.landmark[263].z])
    face_normal = np.cross(right_eye - nose, left_eye - nose)
    face_normal /= np.linalg.norm(face_normal)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            rotation_matrix = Rotation.align_vectors([[0, 0, -1]], [face_normal])[0].as_matrix()
    except Exception as e:
        print(f"Error in head pose estimation: {e}")
        rotation_matrix = np.eye(3)
    return rotation_matrix

def correct_gaze_vector(gaze_vector, head_rotation):
    corrected_gaze = np.dot(head_rotation, gaze_vector)
    return corrected_gaze

def filter_sudden_changes(new_gaze, prev_gaze, max_change_x=15, max_change_y=15):
    if prev_gaze is None:
        return new_gaze
    change_x = abs(new_gaze[0] - prev_gaze[0])
    change_y = abs(new_gaze[1] - prev_gaze[1])
    if change_x > max_change_x:
        new_gaze[0] = prev_gaze[0] + (new_gaze[0] - prev_gaze[0]) * (max_change_x / change_x)
    if change_y > max_change_y:
        new_gaze[1] = prev_gaze[1] + (new_gaze[1] - prev_gaze[1]) * (max_change_y / change_y)
    return new_gaze

def limit_gaze(gaze_point_x, gaze_point_y, screen_width, screen_height):
    gaze_point_x = min(max(gaze_point_x, 0), screen_width - 1)
    gaze_point_y = min(max(gaze_point_y, 0), screen_height - 1)
    return gaze_point_x, gaze_point_y

def calculate_combined_gaze(left_gaze, right_gaze, head_rotation, distance):
    combined_gaze = (left_gaze + right_gaze) / 2
    head_rotation_euler = Rotation.from_matrix(head_rotation).as_euler('xyz')
    return np.concatenate([combined_gaze, head_rotation_euler, [distance]])

def calc(gaze_points, saliency_map):
    count = 0
    total_frames = len(gaze_points)
    logger.info(f"cal -> gaze_points {gaze_points}")
    for frame_num, gaze_point in gaze_points.items():
        x, y = gaze_point
        for saliency_per_frame in saliency_map:
            if int(frame_num) == saliency_per_frame[0]:
                if saliency_per_frame[1][y][x] >= 0.7:
                    count += 1
                    break
    res = count / total_frames if total_frames > 0 else 0
    if res > 0:
        res = round(res, 4) * 100
    return res
################################# Send result #################################
def send_result(final_result, video_id, ip_address):
    data = {
        "video_id": video_id,
        "final_score": final_result,
        "ip_address": ip_address,
        "model_type": "gaze"
    }
    response = requests.post(AGGREGATOR_URL, json=data)

    if response.status_code != 200:
        print(f"Failed to send result: {response.text}")
################################## main code ##################################
@app.route('/gaze', methods=['POST'])
def process_frame():
    data = request.json

    session_id = data['session_id']
    video_id = data['video_id']
    frame_num = data['frame_number']
    last_frame = data['last_frame']
    ip_address = data['ip_address']

    gaze_session = get_session(session_id)

    if not last_frame: 
        frame_base64 = data['frame']
        frame_data = base64.b64decode(frame_base64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = process_single_frame(frame, gaze_session)

        logger.info(f"Frame {frame_num} processed. Result: {result}")
        
        if result:
            gaze_session.final_result[frame_num] = result

        update_session(session_id, gaze_session)
        return jsonify({"status": "success", "message": "Frame processed"}), 200

    else:
        saliency_map = extract_saliencyMap(video_id)
        start_time = time.time()
        timeout = 5 
        try:
            while time.time() - start_time <= timeout:
                final_session = get_session(session_id)
                if len(final_session.final_result) == frame_num:
                    final_res = calc(final_session.final_result, saliency_map)
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("Not all frames processed within 5 seconds")
        
        except TimeoutError as e:
            logger.warning(f"Timeout occurred: {e}. Proceeding with available data.")
            final_session = get_session(session_id)
            final_res = calc(final_session.final_result, saliency_map)

        logger.info(f"Final result: {final_res}")
        logger.info("Sending gaze data to aggregator")
        send_result(final_res, video_id, ip_address)
        
        return jsonify({"status": "success", "message": "Video processing completed"}), 200

def process_single_frame(frame, gaze_session):
    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    h, w = image.shape[:2]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_iris = get_center(face_landmarks.landmark[468:474])[:3]
            right_iris = get_center(face_landmarks.landmark[473:479])[:3]
            left_eye = get_center(face_landmarks.landmark[33:42])[:3]
            right_eye = get_center(face_landmarks.landmark[263:272])[:3]

            estimated_distance = calculate_distance([left_iris, right_iris], image.shape[0])

            left_gaze = estimate_gaze(left_eye, left_iris, estimated_distance)
            right_gaze = estimate_gaze(right_eye, right_iris, estimated_distance)

            head_rotation = estimate_head_pose(face_landmarks)

            left_gaze_corrected = correct_gaze_vector(left_gaze, head_rotation)
            right_gaze_corrected = correct_gaze_vector(right_gaze, head_rotation)

            combined_gaze = calculate_combined_gaze(left_gaze_corrected, right_gaze_corrected, head_rotation, estimated_distance)

            gaze_session.gaze_sequence.append(combined_gaze)

            if len(gaze_session.gaze_sequence) > gaze_session.sequence_length:
                gaze_session.gaze_sequence.pop(0)

            if len(gaze_session.gaze_sequence) == gaze_session.sequence_length:
                gaze_input = np.array(gaze_session.gaze_sequence).flatten().reshape(1, -1)
                with model_lock:
                    predicted_x = model_x.predict(gaze_input)[0]
                    predicted_y = model_y.predict(gaze_input)[0]
                predicted_gaze = np.array([predicted_x, predicted_y])

                gaze_session.gaze_buffer.add(predicted_gaze)
                smoothed_gaze = gaze_session.gaze_buffer.get_average()

                filtered_gaze = filter_sudden_changes(smoothed_gaze, gaze_session.prev_gaze)
                gaze_session.prev_gaze = filtered_gaze

                is_fixation = gaze_session.gaze_fixation.update(filtered_gaze)

                screen_x, screen_y = (filtered_gaze + 1) * np.array([w, h]) / 2
                screen_x, screen_y = limit_gaze(screen_x, screen_y, w, h)

                return int(screen_x), int(screen_y)

    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)