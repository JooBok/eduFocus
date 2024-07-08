import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
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

cv2.namedWindow('MediaPipe Iris Gaze Calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MediaPipe Iris Gaze Calibration', 1280, 720)

class GazeCalibration:
    def __init__(self, degree=2, alpha=0.8):
        self.calibration_points = {
            'q': (-1, 1),   # 좌상단
            'e': (1, 1),    # 우상단
            's': (0, 0),    # 중앙
            'z': (-1, -1),  # 좌하단
            'c': (1, -1)    # 우하단
        }
        self.collected_data = {key: [] for key in self.calibration_points}
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
        self.is_calibrated = False

    def collect_data(self, gaze_point, key):
        self.collected_data[key].append(gaze_point)

    def train_and_save_model(self):
        X = []
        y_x = []
        y_y = []
        for key, points in self.collected_data.items():
            X.extend(points)
            y_x.extend([self.calibration_points[key][0]] * len(points))
            y_y.extend([self.calibration_points[key][1]] * len(points))

        if len(X) > 0:
            X = np.array(X)
            y_x = np.array(y_x)
            y_y = np.array(y_y)

            self.model_x.fit(X, y_x)
            self.model_y.fit(X, y_y)
            joblib.dump(self.model_x, 'model_x.pkl')
            joblib.dump(self.model_y, 'model_y.pkl')
            print("Calibration models saved")
            self.is_calibrated = True
        else:
            print("Not enough data to train the model")

    def predict(self, gaze_point):
        if self.is_calibrated:
            x = self.model_x.predict([gaze_point])[0]
            y = self.model_y.predict([gaze_point])[0]
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

            if calibration.is_calibrated:
                predicted_gaze = calibration.predict(combined_gaze)
                if predicted_gaze:
                    x, y = predicted_gaze
                    screen_x = int((x + 1) * w / 2)
                    screen_y = int((1 - y) * h / 2)
                    cv2.circle(image, (screen_x, screen_y), 10, (0, 255, 0), -1)


########################################################################################## 텍스트 ##########################################################################################
    if not calibration.is_calibrated:
        cv2.putText(image, "Press Q, E, S, Z, C to collect data for calibration", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Q E = Top(L, R), S = middle, Z C = bottom(L, R)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Press G to train and save the model", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(image, "Calibration complete. Tracking gaze...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(image, "Press ESC to exit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('MediaPipe Iris Gaze Calibration', image)
    
    key = cv2.waitKey(1)
    if key & 0xFF in [ord('q'), ord('e'), ord('s'), ord('z'), ord('c')]:
        calibration.collect_data(combined_gaze, chr(key & 0xFF))
        print(f"Collected data for point {chr(key & 0xFF)}")

    if key & 0xFF == ord('g') and not calibration.is_calibrated:
        calibration.train_and_save_model()

    if key & 0xFF == 27:  # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()