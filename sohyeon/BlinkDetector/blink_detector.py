import time, json, logging
import requests, base64, cv2
import mediapipe as mp
import numpy as np
from threading import Lock
from flask import Flask, request, jsonify

import redis, pickle

app = Flask(__name__)
redis_client = redis.Redis(host='redis-service', port=6379, db=2)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#####################################################################################################################################
class BlinkDetector:
    def __init__(self, ear_threshold=0.25, consecutive_frames=3,
                 min_detection_confidence=0.7, min_presence_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_face_mesh = None # 나중에 초기화
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.frame_counter = 0
        self.blink_count = 0
        self.ear_sequence = []
        self.sequence_length = 1200
        self.start_frame = 0
        self.last_left_ear = None
        self.last_right_ear = None
        self.total_frames = 0
        self.minute_blinks = 0
        self.minute_concentration = []
        self.max_ear = 0.3  # 초기 최대 EAR 예시 값

    def initialize_mp_face_mesh(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
 
    def to_dict(self):
        return {
            "min_detection_confidence": self.min_detection_confidence,
            "min_presence_confidence": self.min_presence_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "ear_threshold": self.ear_threshold,
            "consecutive_frames": self.consecutive_frames,
            "frame_counter": self.frame_counter,
            "blink_count": self.blink_count,
            "ear_sequence": self.ear_sequence,
            "sequence_length": self.sequence_length,
            "start_frame": self.start_frame,
            "last_left_ear": self.last_left_ear,
            "last_right_ear": self.last_right_ear,
            "total_frames": self.total_frames,
            "minute_blinks": self.minute_blinks,
            "minute_concentration": self.minute_concentration,
            "max_ear": self.max_ear
        }
    
    @staticmethod
    def from_dict(data):
        detector = BlinkDetector(
            ear_threshold=data["ear_threshold"],
            consecutive_frames=data["consecutive_frames"],
            min_detection_confidence=data["min_detection_confidence"],
            min_presence_confidence=data["min_presence_confidence"],
            min_tracking_confidence=data["min_tracking_confidence"]
        )
        detector.frame_counter = data["frame_counter"]
        detector.blink_count = data["blink_count"]
        detector.ear_sequence = data["ear_sequence"]
        detector.sequence_length = data["sequence_length"]
        detector.start_frame = data["start_frame"]
        detector.last_left_ear = data["last_left_ear"]
        detector.last_right_ear = data["last_right_ear"]
        detector.total_frames = data["total_frames"]
        detector.minute_blinks = data["minute_blinks"]
        detector.minute_concentration = data["minute_concentration"]
        detector.max_ear = data["max_ear"]
        detector.initialize_mp_face_mesh()
        return detector

    def calculate_ear(self, eye_landmarks):
        eye_landmarks = np.array(eye_landmarks)
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        if h == 0:
            return 0
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def calculate_blink_value(self, ear):
        return ear
    
    def update_max_ear(self, ear):
        # 눈이 완전히 열려 있을 때의 EAR 값 갱신
        if ear > self.max_ear:
            self.max_ear = ear

    def detect_blink(self, frame):
        # 프레임에서 얼굴의 눈 깜빡임을 감지
        results = self.mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.total_frames += 1 # 처리된 총 프레임 수 증가

        if not results.multi_face_landmarks:
            logging.debug("No face landmarks detected.")
            return self.last_left_ear, self.last_right_ear, self.blink_count, 0.0, None
        
        
        face_landmarks = results.multi_face_landmarks[0].landmark
        logging.debug(f"Face landmarks detected: {len(face_landmarks)}")
        
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        
        left_eye_landmarks = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in left_eye])
        right_eye_landmarks = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in right_eye])
        
        left_ear = self.calculate_ear(left_eye_landmarks)
        right_ear = self.calculate_ear(right_eye_landmarks)
        logging.debug(f"Left EAR: {left_ear}, Right EAR: {right_ear}")

        e_l = self.calculate_blink_value(left_ear)
        e_r = self.calculate_blink_value(right_ear)

        detection_confidence = self.min_detection_confidence
        presence_confidence = self.min_presence_confidence
        tracking_confidence = self.min_tracking_confidence

        c_l = (detection_confidence + presence_confidence + tracking_confidence) / 3
        c_r = (detection_confidence + presence_confidence + tracking_confidence) / 3
        
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consecutive_frames:
                self.blink_count += 1
                self.minute_blinks += 1
            self.frame_counter = 0

        # 집중도 계산
        concentration = self.calculate_concentration(e_l, e_r, c_l, c_r, self.total_frames)
        logging.debug(f"Concentration calculated: {concentration}")
        self.minute_concentration.append(concentration)

        if len(self.minute_concentration) >= self.sequence_length:  
            self.minute_concentration = []
            self.minute_blinks = 0

        self.last_left_ear = left_ear
        self.last_right_ear = right_ear

        return left_ear, right_ear, self.blink_count, concentration, face_landmarks

    def calculate_concentration(self, e_l, e_r, c_l, c_r, frame_number):
        e_l_normalized = max(0, min(e_l, 1))
        e_r_normalized = max(0, min(e_r, 1))

        if (c_l + c_r) > 0:
            e_c = (c_l * e_l_normalized + c_r * e_r_normalized) / (c_l + c_r)
        else:
            e_c = (e_l_normalized + e_r_normalized) / 2

        if e_c == 0:
            concentration = 0
        else:
            concentration = min(1, 1 / e_c)

        blink_factor = max(0.5, 1 - min(1, self.blink_count / 100))  # 최소값을 0.5로 제한
        concentration *= blink_factor
        
        # ear_sequence에 집중도 추가
        self.ear_sequence.append(concentration)
        if len(self.ear_sequence) > self.sequence_length:
            self.ear_sequence.pop(0)
        
        # 시간에 따른 가중치 적용
        return self.calculate_concentration_score_with_time_weight(frame_number)

    def calculate_concentration_score_with_time_weight(self, frame_number):
        if not self.ear_sequence:
            return 0
        
        # 가중치 계산
        weights = [self.calculate_time_weight(i, frame_number) for i in range(len(self.ear_sequence))]
        weight_sum = sum(weights)
        
        if weight_sum == 0:
            return 0

        # 가중치 적용한 집중도 점수 계산
        concentration_score = sum(ec * w for ec, w in zip(self.ear_sequence, weights)) / weight_sum

        # 집중도 점수를 0과 1 사이로 제한
        normalized_score = max(0, min(concentration_score, 1.0))

        return normalized_score 

    def calculate_time_weight(self, index, frame_number):
        # 시간에 따른 가중치 계산
        # index: EAR 시퀀스의 인덱스
        m = len(self.ear_sequence)
        if m == 0:
            return 0

        max_frames = min(frame_number, self.sequence_length)
        weight = max(0.01, min(1, (m - index) / m * (self.sequence_length - max_frames) / self.sequence_length))
        
        return weight
    

######################################## 결과 전송 ############################################################
AGGREGATOR_URL = "http://result-aggregator-service/aggregate"

def send_result(final_concentration_score, video_id, ip_address):
    data = {
        "video_id": video_id,
        "final_concentration_score": final_concentration_score,
        "ip_address": ip_address,
        "model_type": "blink_detection"
    }
    response = requests.post(AGGREGATOR_URL, json=data)

    if response.status_code != 200:
        print(f"Failed to send result: {response.text}")


######################################### 세션 관리 #############################################################
sessions = {}

class Session:
    def __init__(self):
        self.blink_detector = BlinkDetector()
        self.blink_detector.initialize_mp_face_mesh()
        self.total_concentration = 0
        self.frame_count = 0
        self.start_frame = 0
        self.end_frame = None
        self.total_concentration_scores = []

    def to_dict(self):
        return {
            "blink_detector": self.blink_detector.to_dict(),
            "total_concentration": self.total_concentration,
            "frame_count": self.frame_count,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "total_concentration_scores": self.total_concentration_scores,
        }

    @staticmethod
    def from_dict(data):
        session = Session()
        session.blink_detector = BlinkDetector.from_dict(data["blink_detector"])
        session.total_concentration = data["total_concentration"]
        session.frame_count = data["frame_count"]
        session.start_frame = data["start_frame"]
        session.end_frame = data["end_frame"]
        session.total_concentration_scores = data["total_concentration_scores"]
        return session

def get_session(session_key):
    session_data = redis_client.get(session_key)
    if session_data:
        return Session.from_dict(pickle.loads(session_data))
    return Session()

def save_session(session_key, session):
    redis_client.set(session_key, pickle.dumps(session.to_dict()))
    
######################################### api 엔드포인트 정의 #########################################################
@app.route('/blink', methods=['POST'])
def blink():
    # 폼 데이터에서 필요한 필드 가져오기
    data = request.json

    video_id = data['video_id']
    ip_address = data['ip_address']
    frame_number = data['frame_number']
    is_last_frame = data['last_frame']

    # 필수 데이터가 없는 경우 에러 반환
    if not video_id or frame_number is None:
        return jsonify({"status": "error", "message": "Missing data"}), 400

    # 파일 데이터에서 프레임 가져오기
    frame_base64 = data['frame']
    frame_data = base64.b64decode(frame_base64)
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 세션 키 생성
    session_key = f"{ip_address}_{video_id}"
    session = get_session(session_key)

    if not is_last_frame:
        print("Processing frame...")

        # 깜빡임 감지
        left_ear, right_ear, blink_count, concentration, face_landmarks = session.blink_detector.detect_blink(frame)
        
        # 각 변수의 값을 로그로 기록
        logging.info(f"left_ear: {left_ear}, right_ear: {right_ear}, blink_count: {blink_count}, concentration: {concentration}, face_landmarks: {face_landmarks}")

        # concentration 값을 로그에 기록
        logging.info(f"====================\nConcentration: {concentration}\n====================")
        
        # 집중도 데이터 업데이트
        if face_landmarks is not None:
            avg_ear = (left_ear + right_ear) / 2.0
            session.total_concentration += concentration
            session.frame_count += 1
            session.total_concentration_scores.append(concentration)

        # 세션 저장
        save_session(session_key, session)
        
        logging.info(f"{ip_address} {video_id} {concentration} run succeed")

        return jsonify({"status": "success", "message": "Frame processed"}), 200

    else:
        print("Processing last frame...")
        # 최종 집중도 계산 및 출력
        session.end_frame = frame_number
        if session.frame_count > 0:
            average_concentration = session.total_concentration / session.frame_count
            concentration_percentage = average_concentration * 100
        else:
            concentration_percentage = 0

        # normalized_concentration 변수 정의
        normalized_concentration = concentration_percentage

        # 시간 가중치 적용 및 최종 집중 점수 계산
        total_minutes = session.frame_count / (20 * 60)
        weightings = np.linspace(1, 0.5, max(1, int(total_minutes)))
        concentration_scores = [normalized_concentration * w for w in weightings]
        final_concentration_score = np.mean(concentration_scores)

        # 최종 결과 전송
        send_result(final_concentration_score, video_id, ip_address)
        redis_client.delete(session_key)
        return jsonify({"status": "success", "message": "Video processing completed", "final_concentration_score": final_concentration_score}), 200
    
    return jsonify({"status": "success", "message": "Frame processed"}), 200



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)