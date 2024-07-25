import cv2
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import numpy as np
import joblib
from scipy.spatial.transform import Rotation
import time
import csv
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

### 모델 Load ###
model_x, model_y = joblib.load('new_gaze_model.pkl')

### 시선 데이터 평활화 ###
class GazeBuffer:
    def __init__(self, buffer_size=2, smoothing_factor=0.3):
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
    def __init__(self, velocity_threshold=0.01, duration=0.1, window_size=2):
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

def filter_sudden_changes(new_gaze, prev_gaze, max_change_x=10, max_change_y=10):
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

def calculate_combined_gaze(left_gaze, right_gaze, head_rotation, distance):
    combined_gaze = (left_gaze + right_gaze) / 2
    head_rotation_euler = Rotation.from_matrix(head_rotation).as_euler('xyz')
    return np.concatenate([combined_gaze, head_rotation_euler, [distance]])

def process_image(image_path, gaze_buffer, gaze_fixation, gaze_sequence, prev_gaze):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, prev_gaze

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

            combined_gaze = calculate_combined_gaze(left_gaze, right_gaze, head_rotation, estimated_distance)

            gaze_sequence.append(combined_gaze)
            if len(gaze_sequence) > sequence_length:
                gaze_sequence.pop(0)

            if len(gaze_sequence) == sequence_length:
                gaze_input = np.array(gaze_sequence).flatten().reshape(1, -1)
                predicted_x = model_x.predict(gaze_input)[0]
                predicted_y = model_y.predict(gaze_input)[0]
                predicted = np.array([predicted_x, predicted_y])

                gaze_buffer.add(predicted)
                smoothed_gaze = gaze_buffer.get_average()

                filtered_gaze = filter_sudden_changes(smoothed_gaze, prev_gaze)
                predicted_x, predicted_y = filtered_gaze
                prev_gaze = filtered_gaze

                screen_x = int((predicted_x + 1) * w / 2)
                screen_y = int((1 - predicted_y) * h / 2)

                screen_x, screen_y = limit_gaze_to_screen(screen_x, screen_y, w, h)
                screen_x, screen_y = int(screen_x), int(screen_y)

                return (screen_x, screen_y), prev_gaze

    return None, prev_gaze

gaze_buffer = GazeBuffer(buffer_size=2, smoothing_factor=0.4)
gaze_fixation = GazeFixation(velocity_threshold=0.01, duration=0.1, window_size=2)
gaze_sequence = []
sequence_length = 10
prev_gaze = None

image_folder = './folder' 
output_csv = 'gaze_csv.csv'

with open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame', 'x', 'y']) 

    for _ in range(1, 801):
        if _ <= 9:
            image_path = os.path.join(image_folder, f'frame-000{_}.png')
        elif _ <= 99:
            image_path = os.path.join(image_folder, f'frame-00{_}.png')
        else:
            image_path = os.path.join(image_folder, f'frame-0{_}.png')            

        if not os.path.exists(image_path):
            print(f"Warning: {image_path} does not exist. Skipping.")
            continue

        result, prev_gaze = process_image(image_path, gaze_buffer, gaze_fixation, gaze_sequence, prev_gaze)

        if result:
            x, y = result
            csv_writer.writerow([_, x, y])
            print(f"Processed frame {_}: Gaze coordinates (x={x}, y={y})")
        else:
            print(f"No face detected in frame {_}")

print(f"Processing complete. Results saved to {output_csv}")