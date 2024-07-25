from typing import List, Dict, Tuple, Optional, Union, Any
import time, joblib, requests, base64, cv2, logging, warnings
import mediapipe as mp
import numpy as np
from threading import Lock
from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation
from pymongo import MongoClient
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGGREGATOR_URL = "http://result-aggregator-service/aggregate"
SESSION_SERVICE_URL = "http://session-service"
gaze_detectors = {}
################################## Mongo setting ##################################
MONGO_URI = 'mongodb://root:root@mongodb:27017/saliency_db?authSource=admin'
MONGO_DB = 'saliency_db'
################################## model setting ##################################
global model_x, model_y
model_x = None
model_y = None
model_lock = Lock()

def load_models():
    global model_x, model_y
    with model_lock:
        if model_x is None or model_y is None:
            model_x, model_y = joblib.load('/app/model/gaze_model.pkl')

load_models()
################################## Mongo setting ##################################
def mongodb_client():
    return MongoClient(MONGO_URI)

def extract_saliencyMap(video_id: str) -> List[List[Union[int, np.ndarray]]]:
    """
    video_id에 해당하는 saliency map 데이터를 MongoDB에서 추출하는 함수

    :param  -> video_id: 비디오 ID
    :return -> frame 번호와 saliency map을 포함하는 2차원 리스트
    """
    client = mongodb_client()
    db = client[MONGO_DB]
    collection = db[video_id]
    
    all_data = collection.find()
    extracted_data = []

    for document in all_data:
        frame_num = document.get('frame_num')
        saliency_map = document.get('saliency_map')
        
        if frame_num is not None and saliency_map is not None:
            extracted_data.append([frame_num, saliency_map])
        else:
            logger.error("Error: 'frame_num' or 'saliency_map' not found in the document")

    return extracted_data
################################### Session ###################################
class GazeDetector:
    def __init__(self, session_id):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.session_id = session_id
        self.gaze_buffer = GazeBuffer()
        self.gaze_fixation = GazeFixation()
        self.gaze_sequence = []
        self.prev_gaze = None
        self.sequence_length = 10

    def process_single_frame(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        단일 프레임에서 시선 위치를 추정하는 함수 (process_frame에서 사용)

        :param  -> frame: 처리할 이미지 프레임
        :return -> 추정된 시선 위치 좌표 (x, y) (얼굴이 감지되지 않은 경우 None)
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        h, w = image_rgb.shape[:2]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_iris = get_center(face_landmarks.landmark[468:474])[:3]
                right_iris = get_center(face_landmarks.landmark[473:479])[:3]
                left_eye = get_center(face_landmarks.landmark[33:42])[:3]
                right_eye = get_center(face_landmarks.landmark[263:272])[:3]

                estimated_distance = calculate_distance([left_iris, right_iris], image_rgb.shape[0])

                left_gaze = estimate_gaze(left_eye, left_iris, estimated_distance)
                right_gaze = estimate_gaze(right_eye, right_iris, estimated_distance)

                head_rotation = estimate_head_pose(face_landmarks)

                combined_gaze = calculate_combined_gaze(left_gaze, right_gaze, head_rotation, estimated_distance)

                self.gaze_sequence.append(combined_gaze)

                if len(self.gaze_sequence) > self.sequence_length:
                    self.gaze_sequence.pop(0)

                if len(self.gaze_sequence) == self.sequence_length:
                    gaze_input = np.array(self.gaze_sequence).flatten().reshape(1, -1)
                    with model_lock:
                        predicted_x = model_x.predict(gaze_input)[0]
                        predicted_y = model_y.predict(gaze_input)[0]
                    predicted_gaze = np.array([predicted_x, predicted_y])
                    
                    self.gaze_buffer.add(predicted_gaze)
                    smoothed_gaze = self.gaze_buffer.get_average()

                    filtered_gaze = filter_sudden_changes(smoothed_gaze, self.prev_gaze)
                    predicted_x, predicted_y = filtered_gaze
                    self.prev_gaze = filtered_gaze

                    screen_x = int((predicted_x + 1) * w / 2)
                    screen_y = int((1 - predicted_y) * h / 2)

                    screen_x, screen_y = limit_gaze(screen_x, screen_y, w, h)

                    self.gaze_fixation.update((screen_x, screen_y))

                    return int(screen_x), int(screen_y)

        return None

    def to_dict(self):
        return {
            'session_id': self.session_id,
            'gaze_buffer': self.gaze_buffer.to_dict(),
            'gaze_fixation': self.gaze_fixation.to_dict(),
            'gaze_sequence': [gaze.tolist() for gaze in self.gaze_sequence],
            'prev_gaze': self.prev_gaze.tolist() if self.prev_gaze is not None else None,
            'sequence_length': self.sequence_length
        }

    @classmethod
    def from_dict(cls, data):
        detector = cls(data['session_id'])
        detector.gaze_buffer = GazeBuffer.from_dict(data.get('gaze_buffer', {}))
        detector.gaze_fixation = GazeFixation.from_dict(data.get('gaze_fixation', {}))
        detector.gaze_sequence = [np.array(gaze) for gaze in data.get('gaze_sequence', [])]
        detector.prev_gaze = np.array(data['prev_gaze']) if data.get('prev_gaze') is not None else None
        detector.sequence_length = data.get('sequence_length', 10)
        return detector
    
def get_or_create_gaze_detector(session_id: str) -> GazeDetector:
    """
    session_id에 대한 GazeDetector를 가져오고 없으면 생성하는 함수
    
    :param  -> session_id
    :return -> GazeDetector 인스턴스(세션 별)
    """
    if session_id not in gaze_detectors:
        gaze_detectors[session_id] = GazeDetector(session_id)
    return gaze_detectors[session_id]

def get_session(session_id: str) -> Dict[str, Any]:
    """
    session_id에 대한 세션 데이터를 api를 통해 가져오는 함수
    
    :param  -> session_id
    :return -> session data dict
    """
    try:
        response = requests.get(f"{SESSION_SERVICE_URL}/get_session/{session_id}")
        logger.debug(f"Get session response: {response.status_code}, {response.text}")
        if response.status_code == 200:
            session_data = response.json()
            gaze_data = session_data['components'].get('gaze_tracking', {})
            return gaze_data
        elif response.status_code == 404:
            logger.warning(f"Session not found. Creating new session for {session_id}")
            create_session(session_id)
            return {}
        else:
            logger.error(f"Failed to get session. Status code: {response.status_code}, Response: {response.text}")
            return {}
    except requests.RequestException as e:
        logger.error(f"Request failed when getting session: {e}")
        return {}

def update_session(session_id: str, frame_number: int, gaze_data: Tuple[int, int]) -> None:
    """
    세션 api를 통하여 data를 update(append)하는 함수

    :param -> session_id
    :param -> frame_number
    :param -> gaze_data: 시선 데이터 (x, y)
    """
    try:
        update_data = {
            "component": "gaze_tracking",
            "data": {
                "components": {
                    "gaze_tracking": {
                        str(frame_number): gaze_data
                    }
                }
            }
        }
        response = requests.put(f"{SESSION_SERVICE_URL}/update_session/{session_id}", json=update_data)
        logger.debug(f"Update session response: {response.status_code}, {response.text}")
        if response.status_code == 200:
            logger.info(f"Successfully updated session for {session_id}")
        elif response.status_code == 404:
            logger.warning(f"Session not found when updating. Creating new session for {session_id}")
            create_session(session_id)
            update_session(session_id, frame_number, gaze_data)
        else:
            logger.error(f"Failed to update session. Status code: {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        logger.error(f"Request failed when updating session: {e}")

def create_session(session_id: str) -> None:
    """
    세션이 없을 경우 새로운 세션을 생성하는 함수
    
    :param -> session_id
    """
    try:        
        create_data = {
            "session_id": session_id,
            "components": {
                "gaze_tracking": {}
            }
        }
        response = requests.post(f"{SESSION_SERVICE_URL}/mk-session", json=create_data)
        logger.debug(f"Create session response: {response.status_code}, {response.text}")
        if response.status_code == 201:
            logger.info(f"Successfully created new session for {session_id}")
        else:
            logger.error(f"Failed to create session. Status code: {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        logger.error(f"Request failed when creating session: {e}")
################################# GBR 모델 인풋 만드는 class, 함수 #################################

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
    def to_dict(self):
        return {
            'buffer': [gaze.tolist() for gaze in self.buffer],
            'buffer_size': self.buffer_size,
            'smoothing_factor': self.smoothing_factor,
            'previous_gaze': self.previous_gaze.tolist() if self.previous_gaze is not None else None
        }
    @classmethod
    def from_dict(cls, data):
        buffer = cls(data.get('buffer_size', 3), data.get('smoothing_factor', 0.3))
        buffer.buffer = [np.array(gaze) for gaze in data.get('buffer', [])]
        buffer.previous_gaze = np.array(data['previous_gaze']) if data.get('previous_gaze') is not None else None
        return buffer

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
    def to_dict(self):
        return {
            'velocity_threshold': self.velocity_threshold,
            'duration': self.duration,
            'window_size': self.window_size,
            'gaze_history': [gaze.tolist() for gaze in self.gaze_history],
            'time_history': self.time_history,
            'start_time': self.start_time
        }
    @classmethod
    def from_dict(cls, data):
        fixation = cls(
            data.get('velocity_threshold', 0.1),
            data.get('duration', 0.2),
            data.get('window_size', 6)
        )
        fixation.gaze_history = [np.array(gaze) for gaze in data.get('gaze_history', [])]
        fixation.time_history = data.get('time_history', [])
        fixation.start_time = data.get('start_time')
        return fixation

def calculate_distance(iris_landmarks: List[np.ndarray], image_height: int) -> float:
    """
    홍채 랜드마크를 기반으로 유클리드 거리를 통해 추정 거리를 계산하는 함수
    
    :param  -> iris_landmarks: 홍채 랜드마크 리스트
    :param  -> image_height
    :return -> 추정 거리
    """
    left_iris, right_iris = iris_landmarks
    distance = np.linalg.norm(np.array(left_iris) - np.array(right_iris))
    estimated_distance = (1 / distance) * image_height
    return estimated_distance

def get_center(landmarks: List[mp.framework.formats.landmark.Landmark]) -> np.ndarray:
    """
    랜드마크들의 중심점을 계산하는 함수

    :param  -> landmarks: 랜드마크 리스트
    :return -> 중심점 좌표 (x, y, z)
    """
    return np.mean([[lm.x, lm.y, lm.z] for lm in landmarks], axis=0)

def estimate_gaze(eye_center: np.ndarray, iris_center: np.ndarray, estimated_distance: float) -> np.ndarray:
    """
    눈 중심과 홍채 중심을 기반으로 시선 방향 벡터를 추정하는 함수
    
    :param  -> eye_center: get_center
    :param  -> iris_center: get_center
    :param  -> estimated_distance: calculate_distance
    :return -> 시선 방향 벡터
    """
    eye_vector = iris_center - eye_center
    gaze_point = eye_center + eye_vector * estimated_distance
    return gaze_point

def estimate_head_pose(face_landmarks: mp.framework.formats.landmark.Landmark) -> np.ndarray:
    """
    얼굴 랜드마크를 기반으로 머리 회전 추정 함수
    
    :param  -> face_landmarks: 얼굴 랜드마크
    :return -> 회전 매트릭스
    """
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

def filter_sudden_changes(new_gaze: np.ndarray, prev_gaze: Optional[np.ndarray], max_change_x: float = 15, max_change_y: float = 15) -> np.ndarray:
    """
    시선 변화 상한을 정하는 함수
    
    :param  -> new_gaze: 새로운 시선 위치
    :param  -> prev_gaze: 이전 시선 위치
    :param  -> max_change_x: x축 최대 변화량
    :param  -> max_change_y: y축 최대 변화량
    :return -> 시선 좌표
    """
    if prev_gaze is None:
        return new_gaze
    change_x = abs(new_gaze[0] - prev_gaze[0])
    change_y = abs(new_gaze[1] - prev_gaze[1])
    if change_x > max_change_x:
        new_gaze[0] = prev_gaze[0] + (new_gaze[0] - prev_gaze[0]) * (max_change_x / change_x)
    if change_y > max_change_y:
        new_gaze[1] = prev_gaze[1] + (new_gaze[1] - prev_gaze[1]) * (max_change_y / change_y)
    return new_gaze

def limit_gaze(gaze_point_x: float, gaze_point_y: float, screen_width: int, screen_height: int) -> Tuple[float, float]:
    """
    시선 위치를 화면 내로 제한하는 함수
    
    :param  -> gaze_point_x: 시선 x 좌표
    :param  -> gaze_point_y: 시선 y 좌표
    :param  -> screen_width
    :param  -> screen_height
    :return -> 시선 좌표
    """
    gaze_point_x = min(max(gaze_point_x, 0), screen_width - 1)
    gaze_point_y = min(max(gaze_point_y, 0), screen_height - 1)
    return gaze_point_x, gaze_point_y

def calculate_combined_gaze(left_gaze: np.ndarray, right_gaze: np.ndarray, head_rotation: np.ndarray, distance: float) -> np.ndarray:
    """
    왼쪽 눈과 오른쪽 눈의 시선, 머리 회전, 거리를 결합하여 최종 모델 입력 데이터 생성 함수
    
    :param  -> left_gaze: 왼쪽 눈 시선 벡터
    :param  -> right_gaze: 오른쪽 눈 시선 벡터
    :param  -> head_rotation: 머리 회전 행렬
    :param  -> distance: 추정 거리
    :return -> 모델 input (feature 7개)
    """
    combined_gaze = (left_gaze + right_gaze) / 2
    head_rotation_euler = Rotation.from_matrix(head_rotation).as_euler('xyz')
    return np.concatenate([combined_gaze, head_rotation_euler, [distance]])

def calc(gaze_points: Dict[str, Tuple[int, int]], saliency_map: List[List[Union[int, np.ndarray]]]) -> float:
    """
    시선 위치와 saliency map을 비교하여 점수를 계산하는 함수
    
    :param  -> gaze_points: 프레임별 시선 위치 딕셔너리
    :param  -> saliency_map: saliency map 데이터
    :return -> 점수(최종 output)
    """
    count = 0
    total_frames = len(gaze_points)
    logger.info(f"calc -> gaze_points {gaze_points}")
    for frame_num, gaze_point in gaze_points.items():
        x, y = gaze_point
        for saliency_per_frame in saliency_map:
            if int(frame_num) == saliency_per_frame[0]:
                saliency_threshold = np.percentile(saliency_per_frame[1], 70)
                if saliency_per_frame[1][y][x] >= saliency_threshold:
                    count += 1
                    break
    res = count / total_frames if total_frames > 0 else 0
    if res > 0:
        res = round(res, 4) * 100
    return res
################################# Send result #################################
def send_result(final_result: float, video_id: str, ip_address: str) -> None:
    """
    최종 output을 aggregator로 전송하는 함수
    
    :param -> final_result: 최종 output
    :param -> video_id
    :param -> ip_address
    """
    data = {
        "video_id": video_id,
        "final_score": final_result,
        "ip_address": ip_address,
        "model_type": "gaze"
    }
    try:
        response = requests.post(AGGREGATOR_URL, json=data)
        response.raise_for_status()
        logger.info("Successfully sent result to aggregator")
    except requests.RequestException as e:
        logger.error(f"Failed to send result: {str(e)}")
################################## main code ##################################
@app.route('/gaze', methods=['POST'])
def process_frame():
    data = request.json
    session_id = data['session_id']
    video_id = data['video_id']
    frame_num = data['frame_number']
    last_frame = data['last_frame']
    ip_address = data['ip_address']

    gaze_detector = get_or_create_gaze_detector(session_id)

    if not last_frame: 
        frame_base64 = data['frame']
        frame_data = base64.b64decode(frame_base64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = gaze_detector.process_single_frame(frame)

        logger.info(f"Frame {frame_num} processed. Result: {result}")
        if result:
            update_session(session_id, frame_num, result)

        return jsonify({"status": "success", "message": "Frame processed"}), 200

    if last_frame:
        saliency_map = extract_saliencyMap(video_id)
        central_session_data = get_session(session_id)
        logger.info(f" central_session_data : {central_session_data}")
        final_res = calc(central_session_data, saliency_map)
        send_result(final_res, video_id, ip_address)
        
        if session_id in gaze_detectors:
            del gaze_detectors[session_id]
        
        return jsonify({"status": "success", "message": "Video processing completed"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
