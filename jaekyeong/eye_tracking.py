### ctrl + f 로 VV 찾으면 변경할 수 있는 변수 찾기 가능 ###
import warnings
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from scipy.spatial.transform import Rotation

mp_face_mesh = mp.solutions.face_mesh
### VV face mash 초기화 ###
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

###################################################### 웹캠 설정 ######################################################
cap = cv2.VideoCapture(0)

### VV 해상도 설정 ###
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def resize_image(image, width=1280, height=720):
    return cv2.resize(image, (width, height))

cv2.namedWindow('MediaPipe Iris Gaze', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MediaPipe Iris Gaze', 1280, 720)
######################################################################################################################

class GazeCalibration:
    def __init__(self):
        ### VV 캘리브레이션 테스트 순서 ###
        self.calibration_points = [
            (0, 0),     # 중앙
            (-1, 1),    # 왼쪽 상단
            (1, 1),     # 오른쪽 상단
            (1, -1),    # 오른쪽 하단
            (-1, -1)    # 왼쪽 하단
        ]
        self.collected_data = []
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        ### VV 캘리브레이션 횟수 ###
        self.samples_per_point = 3
        ### 변수 초기화 ###
        self.current_samples = 0
        self.is_calibrated = False

    ############################ 검사 횟수 설정 ############################
    def collect_data(self, gaze_point):
        self.collected_data.append(gaze_point)
        self.current_samples += 1

    def is_point_complete(self):
        return self.current_samples >= self.samples_per_point

    def reset_current_point(self):
        self.current_samples = 0
    #######################################################################

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

    def transform_gaze(self, gaze_point):
        if not self.is_calibrated:
            print("캘리브레이션이 완료되지 않았습니다.")
            return None
        gaze_point = np.array([gaze_point])
        x = self.model_x.predict(gaze_point)[0]
        y = self.model_y.predict(gaze_point)[0]
        return x, y

### 시선 데이터 평활화 ###
class GazeBuffer:
    """
    VV
    안정적인 결과를 얻기 위하여 이전 프레임 시선 데이터와의 연계
    buffer_size: 몇 프레임 전까지 저장하여 평균을 낼 것인가를 정함
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
    def init(self, velocity_threshold=0.1, duration=0.4, window_size=3):
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

### 홍채 중심점의 유클리드 거리를 통해 사용자와 화면 거리 계산 ###
def calculate_distance(iris_landmarks, image_height):
    left_iris, right_iris = iris_landmarks
    distance = np.linalg.norm(np.array(left_iris) - np.array(right_iris))
    estimated_distance = (1 / distance) * image_height
    return estimated_distance

def get_center(landmarks):
    return np.mean([[lm.x, lm.y, lm.z] for lm in landmarks], axis=0)

def draw_point(image, point, color=(0, 255, 0)):
    cv2.circle(image, (int(point[0]), int(point[1])), 5, color, -1)

def estimate_gaze(eye_center, iris_center, estimated_distance):
    eye_vector = iris_center - eye_center
    gaze_point = eye_center + eye_vector * estimated_distance
    return gaze_point

def calculate_combined_gaze(left_gaze, right_gaze):
    return (left_gaze + right_gaze) / 2

######################################################## 고개 회전에 따른 시선의 변화 ########################################################
def estimate_head_pose(face_landmarks):
    ### 얼굴의 주요 랜드마크 선택 ###
    nose = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y, face_landmarks.landmark[1].z])
    left_eye = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y, face_landmarks.landmark[33].z])
    right_eye = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y, face_landmarks.landmark[263].z])
    
    ### 얼굴 평면의 법선 벡터 계산 ###
    face_normal = np.cross(right_eye - nose, left_eye - nose)
    face_normal /= np.linalg.norm(face_normal)
    
    ### 회전 행렬 계산 ###
    rotation_matrix = Rotation.align_vectors([[0, 0, -1]], [face_normal])[0].as_matrix()
    
    return rotation_matrix

def correct_gaze_vector(gaze_vector, head_rotation):
    ### 시선 벡터를 머리 회전에 맞게 보정 ###
    corrected_gaze = np.dot(head_rotation, gaze_vector)
    return corrected_gaze
##############################################################################################################################################

### 노이즈 필터링 ###
def filter_sudden_changes(new_gaze, prev_gaze, max_change=30):
    """
    VV
    max_change: 시선 데이터가 이전 프레임과 비교하여 한 프레임에서 다음 프레임으로 얼마나 많이 변화할 수 있는지를 제한하는 최대 허용 변화량
    """
    if prev_gaze is None:
        return new_gaze
    change = np.linalg.norm(new_gaze - prev_gaze)
    if change > max_change:
        return prev_gaze + (new_gaze - prev_gaze) * (max_change / change)
    return new_gaze

### 시선을 화면 안쪽으로 제한 ###
def limit_gaze_to_screen(gaze_point_x, gaze_point_y, screen_width, screen_height):
    gaze_point_x = min(max(gaze_point_x, 0), screen_width - 1)
    gaze_point_y = min(max(gaze_point_y, 0), screen_height - 1)
    return gaze_point_x, gaze_point_y

# 캘리브레이션 객체 생성
calibration = GazeCalibration()
gaze_buffer = GazeBuffer()
gaze_fixation = GazeFixation()
calibration_index = 0
is_calibrated = False
prev_gaze = None    

# FPS 계산을 위한 변수
prev_frame_time = 0
new_frame_time = 0

# 체크 간격 설정
calibration_check_interval = 1.5
normal_check_interval = 0.05

last_check_time = time.time()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        break

    # FPS 계산
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 이미지를 BGR에서 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    h, w = image.shape[:2]

    current_time = time.time()
    if not is_calibrated:
        if current_time - last_check_time >= calibration_check_interval:
            last_check_time = current_time

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 랜드마크 그리기
                    mp_drawing.draw_landmarks(
                        image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))
                    
                    # # 홍채 랜드마크 추출
                    # left_iris = get_center(face_landmarks.landmark[468:473])
                    # right_iris = get_center(face_landmarks.landmark[473:478])
                    # # 눈 랜드마크 추출
                    # left_eye = get_center(face_landmarks.landmark[33:42])
                    # right_eye = get_center(face_landmarks.landmark[263:272])
                    
                    # 랜드마크 인덱스 수정
                    left_iris = np.mean([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[468:474]], axis=0)[:3]
                    right_iris = np.mean([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[473:479]], axis=0)[:3]
                    left_eye = np.mean([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[33:42]], axis=0)[:3]
                    right_eye = np.mean([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[263:272]], axis=0)[:3]

                    # 거리 계산
                    estimated_distance = calculate_distance([left_iris, right_iris], image.shape[0])

                    # 시선 추정
                    left_gaze = estimate_gaze(left_eye, left_iris, estimated_distance)
                    right_gaze = estimate_gaze(right_eye, right_iris, estimated_distance)

                    # 머리 포즈 추정
                    head_rotation = estimate_head_pose(face_landmarks)

                    left_gaze_corrected = correct_gaze_vector(left_gaze, head_rotation)
                    right_gaze_corrected = correct_gaze_vector(right_gaze, head_rotation)

                    # 통합된 시선 계산 (보정된 벡터 사용)
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
                        else:
                            print("Calibration failed, resetting")
                            calibration_index = 0
                            calibration.collected_data = []

    else:
        if is_calibrated and current_time - last_check_time >= normal_check_interval:
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

                    combined_gaze = (left_gaze_corrected + right_gaze_corrected) / 2

                    ### 시선데이터 평활화 ###
                    gaze_buffer.add(combined_gaze)
                    smoothed_gaze = gaze_buffer.get_average()

                    ### 노이즈 필터링 ###
                    filtered_gaze = filter_sudden_changes(smoothed_gaze, prev_gaze)
                    prev_gaze = filtered_gaze

                    if calibration.is_calibrated:
                        normalized_gaze = calibration.transform_gaze(filtered_gaze)
                        if normalized_gaze is not None:
                            screen_x = int((normalized_gaze[0] + 1) * w / 2)
                            screen_y = int((-normalized_gaze[1] + 1) * h / 2)
                            screen_x, screen_y = limit_gaze_to_screen(screen_x, screen_y, w, h)
                            screen_x, screen_y = screen_x * 2, screen_y * 0.8

                            draw_point(image, (screen_x, screen_y), (255, 255, 0))
                            print(f"Calibrated gaze point: ({screen_x}, {screen_y})")

                            is_fixed = gaze_fixation.update((screen_x, screen_y))
                            if is_fixed:
                                print("Gaze is fixed")
                    else:
                        print("캘리브레이션이 완료되지 않았습니다.")
            else:
                print(f"얼굴인식 불가")

    cv2.imshow('MediaPipe Iris Gaze', image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()