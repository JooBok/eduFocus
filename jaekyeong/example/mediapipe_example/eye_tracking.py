import cv2
import mediapipe as mp
import numpy as np
import joblib
from scipy.spatial.transform import Rotation
import time
import json
from kafka import KafkaConsumer
import asyncio
import aiohttp
from pymongo import MongoClient

MONGO_URI = 'mongodb://mongodb-service:27017'
MONGO_DB = 'database name'
MONGO_COLLECTION = 'saliency_maps'

def get_mongodb_client():
    return MongoClient(MONGO_URI)

def get_saliency_map_from_mongodb(video_id):
    client = get_mongodb_client()
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]
    
    saliency_map_doc = collection.find_one({'video_id': video_id})
    
    if saliency_map_doc:
        return np.array(saliency_map_doc['saliency_map'])
    else:
        raise ValueError(f"Error occured {video_id}")

################################## 상황에 맞게 구현해야 하는 함수 ##################################

### video stream의 처음 메시지에서 video id를 추출하는 함수 ###
def extract_video_id(message):
    pass

### video end 체크 ###
def video_end(message):
    pass

####################################################################################################

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

### model Load ###
model_x = joblib.load('model_x.pkl')
model_y = joblib.load('model_y.pkl')

class GazeBuffer:
    def __init__(self, buffer_size=3):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, gaze):
        self.buffer.append(gaze)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def get_average(self):
        if not self.buffer:
            return None
        return np.mean(self.buffer, axis=0)

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

gaze_buffer = GazeBuffer()
gaze_fixation = GazeFixation()
gaze_sequence = []
sequence_length = 10
prev_gaze = None

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
    if np.linalg.norm(face_normal) > 1e-6:
        face_normal = face_normal / np.linalg.norm(face_normal)
        rotation_matrix = Rotation.align_vectors([[0, 0, -1]], [face_normal])[0].as_matrix()
    else:
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

def limit_gaze_to_screen(gaze_point_x, gaze_point_y, screen_width, screen_height):
    gaze_point_x = min(max(gaze_point_x, 0), screen_width - 1)
    gaze_point_y = min(max(gaze_point_y, 0), screen_height - 1)
    return gaze_point_x, gaze_point_y

### 비동기 frame 처리 ###
async def process_frame(frame, frame_number):
    global gaze_buffer, gaze_fixation, gaze_sequence, prev_gaze

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

            combined_gaze = (left_gaze_corrected + right_gaze_corrected) / 2

            gaze_sequence.append(combined_gaze)
            if len(gaze_sequence) > sequence_length:
                gaze_sequence.pop(0)

            if len(gaze_sequence) == sequence_length:
                gaze_input = np.array(gaze_sequence).flatten().reshape(1, -1)
                predicted_x = model_x.predict(gaze_input)[0]
                predicted_y = model_y.predict(gaze_input)[0]

                gaze_buffer.add(np.array([predicted_x, predicted_y]))
                smoothed_gaze = gaze_buffer.get_average()

                filtered_gaze = filter_sudden_changes(smoothed_gaze, prev_gaze)

                predicted_x, predicted_y = filtered_gaze
                prev_gaze = filtered_gaze

                screen_x = int((predicted_x + 1) * w / 2)
                screen_y = int((1 - predicted_y) * h / 2)

                screen_x, screen_y = limit_gaze_to_screen(screen_x, screen_y, w, h)
                screen_x, screen_y = int(screen_x), int(screen_y)

                return {
                    "frame_number": frame_number,
                    "gaze_point": [screen_x, screen_y]
                }

    return None

### result-aggregator 엔드포인트에 결과값 제출 ###
async def send_result(session, result):
    async with session.post("http://result-aggregator-service/aggregate", json=result) as response:
        return await response.text()

### 결과값 계산 함수 ###
def calc(temp_data, saliency_map):
    count = 0
    total_frames = len(temp_data)
    for frame_data in temp_data.values():
        x, y = frame_data
        if saliency_map[y][x] > 0.7:
            count += 1
    res = count / total_frames
    return res

### kafka producer에서 토픽을 통해 frame 가져오기 ###
async def process_stream():
    consumer = KafkaConsumer('kid_face_video', bootstrap_servers=['kafka-service:9092'])
    
    temp_data = {}
    frame_number = 0
    video_id = None
    
    async with aiohttp.ClientSession() as session:
        for message in consumer:
            
            ### 처음 메시지에서 video의 id를 추출 ###
            if frame_number == 0:
                video_id = extract_video_id(message)
            
            frame = cv2.imdecode(np.frombuffer(message.value, np.uint8), cv2.IMREAD_COLOR)
            result = await process_frame(frame, frame_number)
            if result:
                temp_data[frame_number] = result['gaze_point']
            frame_number += 1

            if is_video_end(message):
                break

        if video_id is None:
            raise ValueError("Could not extract video_id from messages")

        saliency_map = get_saliency_map_from_mongodb(video_id)
        final_result = calc(temp_data, saliency_map)

        await send_result(session, {"final_score": final_result, "video_id": video_id})

asyncio.run(process_stream())
