import warnings
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial.transform import Rotation
import joblib

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
    )
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('MediaPipe Iris Gaze Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MediaPipe Iris Gaze Tracking', 1280, 720)

model_x = joblib.load('model_x.pkl')
model_y = joblib.load('model_y.pkl')

### 시선 데이터 평활화 ###
class GazeBuffer:
    """
    VV
    안정적인 결과를 얻기 위하여 이전 프레임 시선 데이터와의 연계
    buffer_size: 몇 프레임 전까지 저장하여 평균을 낼 것인가를 정함
    """
    def __init__(self, buffer_size=6):
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
    """
    VV
    시선좌표와 시간데이터를 저장하여 속도를 측정하여 fixation 설정
    velocity_threshold: 평균 속도가 이 값보다 낮을 때, 시선이 고정된 것으로 간주 (픽셀/s)
    duration: 시선이 고정된 상태로 인식되기 위해 요구되는 최소 지속 시간
    window_size: 이전 프레임과의 연계 (프레임 개수)
    """
    def __init__(self, velocity_threshold=0.1, duration=0.2, window_size=5):
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

def calculate_combined_gaze(left_gaze, right_gaze):
    return (left_gaze + right_gaze) / 2

def estimate_head_pose(face_landmarks):
    nose = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y, face_landmarks.landmark[1].z])
    left_eye = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y, face_landmarks.landmark[33].z])
    right_eye = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y, face_landmarks.landmark[263].z])
    face_normal = np.cross(right_eye - nose, left_eye - nose)
    face_normal /= np.linalg.norm(face_normal)
    rotation_matrix = Rotation.align_vectors([[0, 0, -1]], [face_normal])[0].as_matrix()
    return rotation_matrix

def correct_gaze_vector(gaze_vector, head_rotation):
    corrected_gaze = np.dot(head_rotation, gaze_vector)
    return corrected_gaze

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


gaze_buffer = GazeBuffer()
gaze_fixation = GazeFixation()
prev_gaze = None    

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    h, w = image.shape[:2]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

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

            combined_gaze = calculate_combined_gaze(left_gaze_corrected, right_gaze_corrected)

            gaze_x = model_x.predict([combined_gaze])[0]
            gaze_y = model_y.predict([combined_gaze])[0]
            
            gaze_buffer.add(np.array([gaze_x, gaze_y]))
            smoothed_gaze = gaze_buffer.get_average()

            ### 노이즈 필터링 ###
            filtered_gaze = filter_sudden_changes(
                new_gaze = smoothed_gaze, 
                prev_gaze = prev_gaze,
                max_change_x = 20,
                max_change_y = 10
                )
            gaze_x, gaze_y = filtered_gaze
            prev_gaze = filtered_gaze

            screen_x = int((gaze_x + 1) * w / 2)
            screen_y = int((1 - gaze_y) * h / 2)

            screen_x, screen_y = limit_gaze_to_screen(screen_x, screen_y, w, h)
            screen_x, screen_y = int(screen_x), int(screen_y)

            cv2.circle(image, (screen_x, screen_y), 10, (0, 255, 0), -1)
            print(f"Calibrated gaze point: ({screen_x}, {screen_y})")           
            
            is_fixed = gaze_fixation.update((screen_x, screen_y))

    cv2.imshow('MediaPipe Iris Gaze Tracking', image)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()