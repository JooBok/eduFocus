import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation

import joblib
import json

### 변경 가능한 변수들은 ctrl + f 에 $를 입력하시면 찾으실 수 있습니다.

### 범용성있는 기초 모델을 만들기 위한 코드입니다.
### 코드 실행시 웹캠 화면이 생성되며 키를 입력하여 데이터를 저장(모델 훈련용) 합니다.
### q,w,e
### a,s,d
### z,x,c 키를 통해 데이터를 저장합니다.
### 데이터 수집이 완료된 후에는 g 키를 입력하여 모델을 훈련시키고 저장시켜 주세요.

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.7, ### $ 최소 탐지 신뢰값 ###
    min_tracking_confidence=0.7   ### $ 최소 추적 신뢰값 ###
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
### $ 해상도 ###
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('MediaPipe Iris Gaze Calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MediaPipe Iris Gaze Calibration', 1280, 720)

class GazeCalibration:
    def __init__(self, sequence_length = 10):
        self.calibration_points = {
            'q': (-1, 1), 'w': (0, 1), 'e': (1, 1),
            'a': (-1, 0), 's': (0, 0), 'd': (1, 0),
            'z': (-1, -1), 'x': (0, -1), 'c': (1, -1)
        }
        self.collected_data = {key: [] for key in self.calibration_points}
        ### $ RF regressor 하이퍼 패러미터 ###
        self.model = RandomForestRegressor(n_estimators = 100, random_state = 777)
        self.is_calibrated = False
        self.sequence_length = sequence_length
    
    def collect_data(self, gaze_point, key):
        self.collected_data[key].append(gaze_point)
    
    ### $ 데이터 증강 ###
    def augment_data(self, X, y, num_augmented = 100):
        augmented_X, augmented_y = [], []
        for _ in range(num_augmented):
            idx = np.random.randint(0, len(X))
            noise = np.random.normal(0, 0.01, X[idx].shape)
            augmented_X.append(X[idx] + noise)
            augmented_y.append(y[idx])
        return np.vstack([X, np.array(augmented_X)]), np.concatenate([y, augmented_y])

    def prepare_sequence(self, data):
        X, y = [], []
        for key, points in data.items():
            if len(points) >= self.sequence_length:
                for i in range(len(points) - self.sequence_length + 1):
                    X.append(np.array(points[i:i+self.sequence_length]).flatten())
                    y.append(self.calibration_points[key])
        return np.array(X), np.array(y)

    def save_data_to_json(self, filename='gaze_data.json', data=None):
        if data is None:
            data = {key: [point.tolist() for point in points] for key, points in self.collected_data.items()}
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Data saved to {filename}")

    def load_data_from_json(self, filename='gaze_data.json'):
        try:
            with open(filename, 'r') as f:
                loaded_data = json.load(f)
            print(f"Data loaded from {filename}")
            return loaded_data
        except FileNotFoundError:
            print(f"File {filename} not found. No previous data loaded.")
            return {}

    def combine_data(self, old_data, new_data):
        combined_data = old_data.copy()
        for key, points in new_data.items():
            if key in combined_data:
                combined_data[key].extend([point.tolist() for point in points])
            else:
                combined_data[key] = [point.tolist() for point in points]
        return combined_data

    def train_and_save_model(self):
        old_data = self.load_data_from_json()
        
        X_old, y_old = self.prepare_sequence(old_data)
        X_new, y_new = self.prepare_sequence(self.collected_data)

        X_combined = np.vstack([X_old, X_new]) if len(X_old) > 0 else X_new
        y_combined = np.vstack([y_old, y_new]) if len(y_old) > 0 else y_new

        if len(X_combined) > 0:
            X_combined, y_combined = self.augment_data(X_combined, y_combined)
            X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=777)

            new_model = RandomForestRegressor(n_estimators=100, random_state=777)
            new_model.fit(X_train, y_train)

            new_score = new_model.score(X_test, y_test)
            print(f"New model R² score: {new_score:.4f}")

            ###################################### 기존모델, 신모델 비교 ######################################
            try:
                old_model = joblib.load('gaze_model.pkl')
                old_score = old_model.score(X_test, y_test)

                print(f"Old model R² score: {old_score:.4f}")

                if new_score > old_score:
                    print("New model performs better. Saving new model and combining data.")
                    self.model = new_model
                    joblib.dump(self.model, 'gaze_model.pkl')
                    combined_data = self.combine_data(old_data, self.collected_data)
                    self.save_data_to_json(data=combined_data)
                else:
                    print("Old model performs better. Keeping old model and old data.")
                    self.model = old_model
                    self.save_data_to_json(data=old_data)
            except FileNotFoundError:
                print("No existing model found. Saving new model and new data.")
                self.model = new_model
                joblib.dump(self.model, 'gaze_model.pkl')
                self.save_data_to_json()

            self.is_calibrated = True
        else:
            print("ERROR | Not enough data to train the model")

    def predict(self, gaze_sequence):
        """
        RF regressor 사용하여 calculate_combined_gaze 함수로 계산한 좌표를 실제 시선 좌표로 predict.
        """
        if self.is_calibrated:
            gaze_sequence = np.array(gaze_sequence).flatten().reshape(1, -1)
            return self.model.predict(gaze_sequence)[0]
        else:
            return None

### 유클리드거리로 화면과 사용자 상대적 거리 계산 ###
def calculate_distance(iris_landmarks, image_height):
    left_iris, right_iris = iris_landmarks
    distance = np.linalg.norm(np.array(left_iris) - np.array(right_iris))
    estimated_distance = (1 / distance) * image_height
    return estimated_distance

### 3차원으로 눈과 홍채 중심 계산(평균) ###
def get_center(landmarks):
    return np.mean([[lm.x, lm.y, lm.z] for lm in landmarks], axis=0)

### 홍채이동 벡터 * calculate_distance -> 시선 이동 거리 ###
def estimate_gaze(eye_center, iris_center, estimated_distance):
    eye_vector = iris_center - eye_center
    gaze_point = eye_center + eye_vector * estimated_distance
    return gaze_point

### 머리 회전 추적 ###
def estimate_head_pose(face_landmarks):
    nose = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y, face_landmarks.landmark[1].z])
    left_eye = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y, face_landmarks.landmark[33].z])
    right_eye = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y, face_landmarks.landmark[263].z])
    face_normal = np.cross(right_eye - nose, left_eye - nose)
    face_normal /= np.linalg.norm(face_normal)
    rotation_matrix = Rotation.align_vectors([[0, 0, -1]], [face_normal])[0].as_matrix()
    return rotation_matrix

### 머리 회전과 시선 벡터 추적 ###
def correct_gaze_vector(gaze_vector, head_rotation):
    corrected_gaze = np.dot(head_rotation, gaze_vector)
    return corrected_gaze

### 왼쪽시선과 오른쪽시선 결합 (평균) ###
def calculate_combined_gaze(left_gaze, right_gaze, head_rotation, distance):
    combined_gaze = (left_gaze + right_gaze) / 2
    head_rotation_euler = Rotation.from_matrix(head_rotation).as_euler('xyz')
    return np.concatenate([combined_gaze, head_rotation_euler, [distance]])

calibration = GazeCalibration()
gaze_sequence = []

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

            combined_gaze = calculate_combined_gaze(left_gaze_corrected, right_gaze_corrected, head_rotation, estimated_distance)

            gaze_sequence.append(combined_gaze)
            if len(gaze_sequence) > calibration.sequence_length:
                gaze_sequence.pop(0)

            if calibration.is_calibrated and len(gaze_sequence) == calibration.sequence_length:
                predicted_gaze = calibration.predict(gaze_sequence)
                if predicted_gaze is not None:
                    x, y = predicted_gaze
                    screen_x = int((x + 1) * w / 2)
                    screen_y = int((1 - y) * h / 2)
                    cv2.circle(image, (screen_x, screen_y), 10, (0, 255, 0), -1)

    ### 화면에 점 표시 코드 ###
    for point, (px, py) in calibration.calibration_points.items():
        screen_x = int((px + 1) * w / 2)
        screen_y = int((1 - py) * h / 2)
        cv2.circle(image, (screen_x, screen_y), 10, (0, 0, 255), -1)
        cv2.putText(image, point, (screen_x - 20, screen_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


########################################################################### 텍스트 ###########################################################################
    if not calibration.is_calibrated:
        cv2.putText(image, "Press Q, W, E, A, S, D, Z, X, C to collect data for calibration", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Q E = Top(L, R), S = middle, Z C = bottom(L, R)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Press G to train and save the model", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(image, "Calibration complete. press ESC and run prediction.py", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(image, "Press ESC to exit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('MediaPipe Iris Gaze Calibration', image)
##############################################################################################################################################################

    ### $ waitkey조절하여 프레임 조절 가능 ###
    key = cv2.waitKey(1)

    if key & 0xFF in [ord('q'), ord('w'), ord('e'), ord('a'), ord('s'), ord('d'), ord('z'), ord('x'), ord('c')]:
        if len(gaze_sequence) == calibration.sequence_length:
            calibration.collect_data(combined_gaze, chr(key & 0xFF))
            print(f"Collected data for point {chr(key & 0xFF)}")

    if key & 0xFF == ord('g') and not calibration.is_calibrated:
        calibration.train_and_save_model()

    ### ESC 키 입력 종료 ###
    if key & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()