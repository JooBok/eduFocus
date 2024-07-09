import warnings
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from scipy.spatial.transform import Rotation
from sklearn.linear_model import LinearRegression
import joblib

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('MediaPipe Iris Gaze Calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MediaPipe Iris Gaze Calibration', 1280, 720)

class GazeCalibration:
    def __init__(self, degree=2, alpha=0.75):
        self.calibration_points = [
            (0, 0),     # 중앙
            (-1, 1),    # 왼쪽 상단
            (1, 1),     # 오른쪽 상단
            (1, -1),    # 오른쪽 하단
            (-1, -1)    # 왼쪽 하단
        ]
        self.collected_data = []

        self.model_x = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree, include_bias=False),
            Ridge(alpha=alpha)
        )
        self.model_y = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree, include_bias=False),
            Ridge(alpha=alpha)
        )

        # self.model_x = LinearRegression()
        # self.model_y = LinearRegression()     
           
        self.samples_per_point = 5
        self.current_samples = 0
        self.is_calibrated = False

    def collect_data(self, gaze_point):
        self.collected_data.append(gaze_point)
        self.current_samples += 1

    def is_point_complete(self):
        return self.current_samples >= self.samples_per_point

    def reset_current_point(self):
        self.current_samples = 0

    def calibrate(self):
        print(f"Attempting calibration with {len(self.collected_data)} samples")
        if len(self.collected_data) == len(self.calibration_points) * self.samples_per_point:
            X = np.array(self.collected_data)
            y_x = np.repeat([point[0] for point in self.calibration_points], self.samples_per_point)
            y_y = np.repeat([point[1] for point in self.calibration_points], self.samples_per_point)

            self.model_x.fit(X, y_x)
            self.model_y.fit(X, y_y)
            self.is_calibrated = True
            print("Calibration completed successfully")
            return True
        else:
            print(f"Not enough data for calibration. Collected: {len(self.collected_data)}, Required: {len(self.calibration_points) * self.samples_per_point}")
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

calibration = GazeCalibration()
calibration_index = 0
is_calibrated = False

calibration_check_interval = 1
last_check_time = time.time()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    h, w = image.shape[:2]

    current_time = time.time()
    if not is_calibrated:
        if current_time - last_check_time >= calibration_check_interval:
            last_check_time = current_time

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

                    left_iris = np.mean([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[468:474]], axis=0)[:3]
                    right_iris = np.mean([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[473:479]], axis=0)[:3]
                    left_eye = np.mean([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[33:42]], axis=0)[:3]
                    right_eye = np.mean([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[263:272]], axis=0)[:3]

                    estimated_distance = calculate_distance([left_iris, right_iris], image.shape[0])

                    left_gaze = estimate_gaze(left_eye, left_iris, estimated_distance)
                    right_gaze = estimate_gaze(right_eye, right_iris, estimated_distance)

                    head_rotation = estimate_head_pose(face_landmarks)

                    left_gaze_corrected = correct_gaze_vector(left_gaze, head_rotation)
                    right_gaze_corrected = correct_gaze_vector(right_gaze, head_rotation)

                    combined_gaze = calculate_combined_gaze(left_gaze_corrected, right_gaze_corrected)

                    if calibration_index < len(calibration.calibration_points):
                        target_point = calibration.calibration_points[calibration_index]
                        x, y = int((target_point[0] + 1) * w / 2), int((-target_point[1] + 1) * h / 2)
                        cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

                        point_names = ["중앙", "왼쪽 상단", "오른쪽 상단", "오른쪽 하단", "왼쪽 하단"]
                        cv2.putText(image, f"Look at {point_names[calibration_index]}", (10, h - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        calibration.collect_data(combined_gaze)
                        print(f"Collected sample {calibration.current_samples}/{calibration.samples_per_point} for point {calibration_index+1}")

                        if calibration.is_point_complete():
                            calibration_index += 1
                            calibration.reset_current_point()
                            print(f"Completed calibration for point {calibration_index}/{len(calibration.calibration_points)}")

                    if calibration_index >= len(calibration.calibration_points):
                        is_calibrated = calibration.calibrate()
                        if is_calibrated:
                            print("Full calibration completed")
                            joblib.dump(calibration.model_x, 'model_x.pkl')
                            joblib.dump(calibration.model_y, 'model_y.pkl')
                            print("Calibration models saved")
                            break
                        else:
                            print("Calibration failed, resetting")
                            calibration_index = 0
                            calibration.collected_data = []

    cv2.imshow('MediaPipe Iris Gaze Calibration', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
