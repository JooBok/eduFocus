import cv2
import mediapipe as mp
import numpy as np
import joblib
from scipy.spatial.transform import Rotation
import time
import json

### 변경 가능한 변수들은 ctrl + f 에 $를 입력하시면 찾으실 수 있습니다.
### RF_reg_calibrate.py에서 저장한 모델을 사용하여 시선 예측하는 파일입니다.

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

### 모델 Load ###
model = joblib.load('gaze_model.pkl')

### 시선 데이터 평활화 ###
class GazeBuffer:
    """
    안정적인 결과를 얻기 위하여 이전 프레임 시선 데이터와의 연계
    $ buffer_size: 몇 프레임 전까지 저장하여 평균을 낼 것인가를 정함
    """
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
    """
    시선좌표와 시간데이터를 저장하여 속도를 측정하여 fixation 설정
    $ velocity_threshold: 평균 속도가 이 값보다 낮을 때, 시선이 고정된 것으로 간주 (픽셀/s)
    $ duration: 시선이 고정된 상태로 인식되기 위해 요구되는 최소 지속 시간
    $ window_size: 이전 프레임과의 연계 (프레임 개수)
    """
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

### $ 프레임당 시선좌표 이동속도 제한 ###
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

### 시선이 화면 밖으로 나가지 않도록 제한하는 함수 ###
def limit_gaze_to_screen(gaze_point_x, gaze_point_y, screen_width, screen_height):
    gaze_point_x = min(max(gaze_point_x, 0), screen_width - 1)
    gaze_point_y = min(max(gaze_point_y, 0), screen_height - 1)
    return gaze_point_x, gaze_point_y

gaze_buffer = GazeBuffer()
gaze_fixation = GazeFixation()
gaze_sequence = []
sequence_length = 10
prev_gaze = None    
frame = 0
temp_data = dict()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    h, w = image.shape[:2]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

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
                ### flatten ###
                gaze_input = np.array(gaze_sequence).flatten().reshape(1, -1)
                ### 모델 예측 ###
                predicted = model.predict(gaze_input)[0]

                ### 평활화 ###
                gaze_buffer.add(np.array(predicted))
                smoothed_gaze = gaze_buffer.get_average()

                ### 노이즈 필터링 ###
                filtered_gaze = filter_sudden_changes(
                    new_gaze = smoothed_gaze, 
                    prev_gaze = prev_gaze
                    )

                predicted_x, predicted_y = filtered_gaze
                ### previous gaze 갱신 ###
                prev_gaze = filtered_gaze

                screen_x = int((predicted_x + 1) * w / 2)
                screen_y = int((1 - predicted_y) * h / 2)

                screen_x, screen_y = limit_gaze_to_screen(screen_x, screen_y, w, h)
                screen_x, screen_y = int(screen_x), int(screen_y)

                cv2.circle(image, (screen_x, screen_y), 10, (0, 255, 0), -1)
                print(f"Calibrated gaze point: ({screen_x}, {screen_y})")    

                ### Fixation ###
                is_fixed = gaze_fixation.update((screen_x, screen_y))
            
                frame += 1

                temp_data[frame] = [screen_x, screen_y]
    
    cv2.imshow('MediaPipe Iris Gaze Prediction', image)

    ### ESC 키 입력 종료 ###
    if cv2.waitKey(1) & 0xFF == 27:
        break
########################################################################
#### {Mongo db} extract saliency map 교체 예정
saliency_json = 'saliency_map.json'

with open(saliency_json, 'r') as f:
    saliency_map_list = json.load(f)
    saliency_map = np.array(saliency_map_list)

########################################################################
def calc(temp_data, saliency_map):
    count = 0
    try:
        for _ in range(1, len(saliency_map) + 1):
            x, y = temp_data[_]

            if saliency_map[y][x] > 0.7:
                count += 1
            else:
                pass
    except:
        pass
    res = count / len(saliency_map)
    return res

result = calc(temp_data, saliency_map)
print(temp_data)
print(f'++++++++++++++++++++++++++++++++++++++++++++{result}')
########################################################################
# aggregator_url = "http://result-aggregator-service/aggregate"

# import requests
# requests.post(aggregator_url, json=result)
########################################################################


cap.release()
cv2.destroyAllWindows()