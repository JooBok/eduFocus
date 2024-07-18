import time, json, joblib
import requests, base64, cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import mediapipe as mp
import numpy as np
from threading import Lock

from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation
from pymongo import MongoClient
import redis, pickle

app = Flask(__name__)

redis_client = redis.Redis(host='redis-service', port=6379, db=0)
AGGREGATOR_URL = "http://result-aggregator-service/aggregate"
################################## Mongo setting ##################################
# MONGO_URI = 'mongodb://mongodb-service:27017'
MONGO_URI = 'mongodb://root:root@mongodb:27017/saliency_db?authSource=admin'
MONGO_DB = 'saliency_db'
MONGO_COLLECTION = 'contents2'
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

# def extract_saliencyMap(video_id):
#     client = mongodb_client()
#     db = client[MONGO_DB]
#     collection = db[MONGO_COLLECTION]
    
#     saliency_map_doc = collection.find_one({'video_id': video_id})
    
#     if saliency_map_doc:
#         return np.array(saliency_map_doc['saliency_map'])
#     else:
#         raise ValueError(f"Error occurred {video_id}")
def extract_saliencyMap(frame_num):
    client = mongodb_client()
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]
    
    # frame_num으로 문서를 찾음
    saliency_map_doc = collection.find_one({'frame_num': frame_num})
    
    if saliency_map_doc:
        return np.array(saliency_map_doc['saliency_map'])
    else:
        raise ValueError(f"Error occurred: frame_num {frame_num} not found")
################################# calc Class, def #################################
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
################################### Session ###################################
sessions = {}

class Session:
    def __init__(self):
        self.gaze_buffer = GazeBuffer()
        self.gaze_fixation = GazeFixation()
        self.gaze_sequence = []
        self.prev_gaze = None
        self.final_result = {}
        self.sequence_length = 10
    def to_dict(self):
        return {
            'gaze_buffer': self.gaze_buffer,
            'gaze_fixation': self.gaze_fixation,
            'gaze_sequence': self.gaze_sequence,
            'prev_gaze': self.prev_gaze,
            'final_result': self.final_result,
            'sequence_length': self.sequence_length
        }
    @classmethod
    def from_dict(cls, data):
        session = cls()
        session.gaze_buffer = data['gaze_buffer']
        session.gaze_fixation = data['gaze_fixation']
        session.gaze_sequence = data['gaze_sequence']
        session.prev_gaze = data['prev_gaze']
        session.final_result = data['final_result']
        session.sequence_length = data['sequence_length']
        return session

def get_session(session_key):
    session_data = redis_client.get(session_key)
    if session_data:
        return Session.from_dict(pickle.loads(session_data))
    return Session()
###############################################################################
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

def calc(final_result, saliency_map):
    count = 0
    total_frames = len(final_result)
    for frame_num, gaze_point in final_result.items():
        x, y = gaze_point
        if saliency_map[frame_num][y][x] >= 0.7:
            count += 1
    res = count / total_frames
    return res

################################# Send result #################################
def send_result(final_result, video_id, ip_address):
    data = {
        "video_id": video_id,
        "final_score": final_result,
        "ip_address": ip_address
    }
    response = requests.post(AGGREGATOR_URL, json=data)

    if response.status_code != 200:
        print(f"Failed to send result: {response.text}")
################################## main code ##################################
@app.route('/gaze', methods=['POST'])
def process_frame():
    data = request.json

    video_id = data['video_id']
    ip_address = data['ip_address']
    frame_num = data['frame_number']
    last_frame = data['last_frame']


    session_key = f"{ip_address}_{video_id}"

    session = get_session(session_key)

    if not last_frame:
        frame_base64 = data['frame']
        frame_data = base64.b64decode(frame_base64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = process_single_frame(frame, session)
        
        if result:
            session.final_result[frame_num] = result['gaze_point']
        
        redis_client.set(session_key, pickle.dumps(session.to_dict()))        
        return jsonify({"status": "success", "message": "Frame processed"}), 200

    else:
        saliency_map = extract_saliencyMap(video_id)
        final_res = calc(session.final_result, saliency_map)
        send_result(final_res, video_id, ip_address)
        
        redis_client.delete(session_key)
        return jsonify({"status": "success", "message": "Video processing completed"}), 200

def process_single_frame(frame, session):
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

            ### 7차원 데이터로 변경 ###
            combined_gaze = calculate_combined_gaze(left_gaze_corrected, right_gaze_corrected, head_rotation, estimated_distance)

            session.gaze_sequence.append(combined_gaze)

            if len(session.gaze_sequence) > session.sequence_length:
                session.gaze_sequence.pop(0)

            if len(session.gaze_sequence) == session.sequence_length:
                gaze_input = np.array(session.gaze_sequence).flatten().reshape(1, -1)
                with model_lock:
                    predicted_x = model_x.predict(gaze_input)[0]
                    predicted_y = model_y.predict(gaze_input)[0]
                predicted_gaze = np.array([predicted_x, predicted_y])

                session.gaze_buffer.add(predicted_gaze)
                smoothed_gaze = session.gaze_buffer.get_average()

                filtered_gaze = filter_sudden_changes(smoothed_gaze, session.prev_gaze)

                predicted_x, predicted_y = filtered_gaze
                session.prev_gaze = filtered_gaze

                screen_x = int((predicted_x + 1) * w / 2)
                screen_y = int((1 - predicted_y) * h / 2)

                screen_x, screen_y = limit_gaze(screen_x, screen_y, w, h)
                screen_x, screen_y = int(screen_x), int(screen_y)

                return {
                    "gaze_point": [screen_x, screen_y]
                }

    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
