import cv2
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow('MediaPipe Iris Gaze Calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MediaPipe Iris Gaze Calibration', 640, 480)

class GazeCalibration:
    def __init__(self, sequence_length = 10):
        self.calibration_points = {
            'q': (-1, 1), 'w': (0, 1), 'e': (1, 1),
            'a': (-1, 0), 's': (0, 0), 'd': (1, 0),
            'z': (-1, -1), 'x': (0, -1), 'c': (1, -1)
        }
        self.collected_data = {key: [] for key in self.calibration_points}
        ### $ RF regressor 하이퍼 패러미터 ###
        self.model_x = None
        self.model_y = None
        self.is_calibrated = False
        self.sequence_length = sequence_length

        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.03, 0.05],
            'max_depth': [3, 4],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 3],
            'subsample': [0.8, 1.0],
            'max_features': ['sqrt', None]
        }

    def collect_data(self, gaze_point, key):
        self.collected_data[key].append(gaze_point)
    
    def augment_data(self, X, y):
        num_augmented = len(X) * 2
        augmented_X, augmented_y = [], []

        for _ in range(num_augmented):
            idx = np.random.randint(0, len(X))
            original_data = X[idx]
            augmented_data = original_data.copy()

            ### 작은 회전 적용 ###
            rotation_angle = np.random.uniform(-5, 5)
            rotation = Rotation.from_euler('xyz', [0, 0, np.radians(rotation_angle)])
            augmented_data[:3] = rotation.apply(augmented_data[:3])

            ### 작은 스케일 변화 적용 ###
            scale_factor = np.random.uniform(0.95, 1.05)
            augmented_data *= scale_factor

            ### 작은 노이즈 추가 ###
            noise = np.random.normal(0, 0.02, augmented_data.shape)
            augmented_data += noise

            augmented_X.append(augmented_data)
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
            X_augmented, y_augmented = self.augment_data(X_combined, y_combined)
            X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=777)

            gb_regressor = GradientBoostingRegressor(random_state=777)
            grid_search = GridSearchCV(estimator=gb_regressor, param_grid=self.param_grid, 
                                    cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

            grid_search.fit(X_train, y_train[:, 0])
            best_model_x = grid_search.best_estimator_

            print("Best parameters for x coordinate:", grid_search.best_params_)
            print("Best score for x coordinate:", -grid_search.best_score_)
        
            grid_search.fit(X_train, y_train[:, 1])
            best_model_y = grid_search.best_estimator_
            
            print("Best parameters for y coordinate:", grid_search.best_params_)
            print("Best score for y coordinate:", -grid_search.best_score_)

            y_pred_x = best_model_x.predict(X_test)
            y_pred_y = best_model_y.predict(X_test)
            
            mse_x = mean_squared_error(y_test[:, 0], y_pred_x)
            mse_y = mean_squared_error(y_test[:, 1], y_pred_y)
            r2_x = r2_score(y_test[:, 0], y_pred_x)
            r2_y = r2_score(y_test[:, 1], y_pred_y)

            print(f"MSE for x: {mse_x}, MSE for y: {mse_y}")
            print(f"R2 score for x: {r2_x}, R2 score for y: {r2_y}")

            new_score = (r2_x + r2_y) / 2

            ###################################### 기존모델, 신모델 비교 ######################################
            try:
                old_model_x, old_model_y = joblib.load('new_gaze_model.pkl')
                old_score_x = old_model_x.score(X_test, y_test[:, 0])
                old_score_y = old_model_y.score(X_test, y_test[:, 1])
                old_score = (old_score_x + old_score_y) / 2

                print(f"Old model R² score: {old_score:.4f}")
                print(f"New model R² score: {new_score:.4f}")

                if new_score > old_score:
                    print("New model performs better. Saving new model and combining data.")
                    self.model_x = best_model_x
                    self.model_y = best_model_y
                    joblib.dump((self.model_x, self.model_y), 'new_gaze_model.pkl')
                    combined_data = self.combine_data(old_data, self.collected_data)
                    self.save_data_to_json(data=combined_data)
                else:
                    print("Old model performs better. Keeping old model and old data.")
                    self.model_x, self.model_y = old_model_x, old_model_y
                    self.save_data_to_json(data=old_data)
            except FileNotFoundError:
                print("No existing model found. Saving new model and new data.")
                self.model_x = best_model_x
                self.model_y = best_model_y
                joblib.dump((self.model_x, self.model_y), 'new_gaze_model.pkl')
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
            x_pred = self.model_x.predict(gaze_sequence)
            y_pred = self.model_y.predict(gaze_sequence)
            return np.array([x_pred[0], y_pred[0]])
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

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            rotation_matrix = Rotation.align_vectors([[0, 0, -1]], [face_normal])[0].as_matrix()
    except Exception as e:
        print(f"Error in head pose estimation: {e}")
        rotation_matrix = np.eye(3)
    return rotation_matrix

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

            combined_gaze = calculate_combined_gaze(left_gaze, right_gaze, head_rotation, estimated_distance)

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
