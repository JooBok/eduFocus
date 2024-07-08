import cv2
import mediapipe as mp
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation

import joblib

### 범용성있는 기초 모델을 만들기 위한 코드입니다.
### 코드 실행시 웹캠 화면이 생성되며 키를 입력하여 데이터를 저장(모델 훈련용) 합니다.
### q,w,e
### a,s,d
### z,x,c 키를 통해 데이터를 저장합니다.
### 데이터 수집이 완료된 후에는 g 키를 입력하여 모델을 훈련시키고 저장시켜 주세요.

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.7, # 
    min_tracking_confidence=0.7   # 
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
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
        self.model_x = RandomForestRegressor(n_estimators = 100, random_state = 777)
        self.model_y = RandomForestRegressor(n_estimators = 100, random_state = 777)
        self.is_calibrated = False
        self.sequence_length = sequence_length
    
    def collect_data(self, gaze_point, key):
        self.collect_data[key].append(gaze_point)
    
    def augment_data(self, X, y, num_augmented = 1000):
        augmented_X, augmented_y = [], []
        for _ in range(num_augmented):
            idx = np.random.randint(0, len(X))
            noise = np.random.nomal(0, 0.01, X[idx].shape)
            augmented_X.append(X[idx] + noise)
            augmented_y.append(y[idx])
        return np.vstack([X, np.array(augmented_X)]), np.concatenate([y, augmented_y])

    def prepare_sequence(self, data):
        X, y = [], []
        for key, points in data.item():
            if len(points) >= self.sequence_length:
                for i in range(len(points) - self.sequence_length + 1):
                    X.append(points[i:i+self.sequence_length])
                    y.append(self.calibration_points[key])
        return np.array(X), np.array(y)

    def train_and_save_model(self):
        X, y = self.prepare_sequences(self.collected_data)
        if len(X) > 0:
            X = X.reshape(X.shape[0], -1)  # Flatten the sequence
            X, y = self.augment_data(X, y)  # 데이터 증강
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.model_x.fit(X_train, y_train[:, 0])
            self.model_y.fit(X_train, y_train[:, 1])

            x_score = self.model_x.score(X_test, y_test[:, 0])
            y_score = self.model_y.score(X_test, y_test[:, 1])
            print(f"Model X R² score: {x_score:.4f}")
            print(f"Model Y R² score: {y_score:.4f}")

            joblib.dump(self.model_x, 'model_x.pkl')
            joblib.dump(self.model_y, 'model_y.pkl')
            print("Calibration models saved")
            self.is_calibrated = True
        else:
            print("Not enough data to train the model")

    def predict(self, gaze_sequence):
        if self.is_calibrated:
            gaze_sequence = np.array(gaze_sequence).flatten().reshape(1, -1)
            x = self.model_x.predict(gaze_sequence)[0]
            y = self.model_y.predict(gaze_sequence)[0]
            return x, y
        else:
            return None

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
    rotation_matrix = Rotation.align_vectors([[0, 0, -1]], [face_normal])[0].as_matrix()
    return rotation_matrix

def correct_gaze_vector(gaze_vector, head_rotation):
    corrected_gaze = np.dot(head_rotation, gaze_vector)
    return corrected_gaze

def calculate_combined_gaze(left_gaze, right_gaze):
    return (left_gaze + right_gaze) / 2

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

            combined_gaze = calculate_combined_gaze(left_gaze_corrected, right_gaze_corrected)

            gaze_sequence.append(combined_gaze)
            if len(gaze_sequence) > calibration.sequence_length:
                gaze_sequence.pop(0)

            if calibration.is_calibrated and len(gaze_sequence) == calibration.sequence_length:
                predicted_gaze = calibration.predict(gaze_sequence)
                if predicted_gaze:
                    x, y = predicted_gaze
                    screen_x = int((x + 1) * w / 2)
                    screen_y = int((1 - y) * h / 2)
                    cv2.circle(image, (screen_x, screen_y), 10, (0, 255, 0), -1)

    # 화면에 항상 calibration_points 좌표에 원을 표시
    for point, (px, py) in calibration.calibration_points.items():
        screen_x = int((px + 1) * w / 2)
        screen_y = int((1 - py) * h / 2)
        cv2.circle(image, (screen_x, screen_y), 10, (0, 0, 255), -1)
        cv2.putText(image, point, (screen_x - 20, screen_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


########################################################################################## 텍스트 ##########################################################################################
    if not calibration.is_calibrated:
        cv2.putText(image, "Press Q, W, E, A, S, D, Z, X, C to collect data for calibration", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Q E = Top(L, R), S = middle, Z C = bottom(L, R)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Press G to train and save the model", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(image, "Calibration complete. press ESC and run prediction.py", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(image, "Press ESC to exit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('MediaPipe Iris Gaze Calibration', image)
    
    key = cv2.waitKey(1)
    if key & 0xFF in [ord('q'), ord('w'), ord('e'), ord('a'), ord('s'), ord('d'), ord('z'), ord('x'), ord('c')]:
        if len(gaze_sequence) == calibration.sequence_length:
            calibration.collect_data(combined_gaze, chr(key & 0xFF))
            print(f"Collected data for point {chr(key & 0xFF)}")

    if key & 0xFF == ord('g') and not calibration.is_calibrated:
        calibration.train_and_save_model()

    if key & 0xFF == 27:  # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()