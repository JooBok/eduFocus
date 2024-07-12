import time, json
import requests, base64, cv2
import mediapipe as mp
import numpy as np
from threading import Lock
from flask import Flask, request, jsonify

app = Flask(__name__)

class BlinkDetector:
    def __init__(self, ear_threshold=0.25, consecutive_frames=3,
                 min_detection_confidence=0.7, min_presence_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
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
            return self.last_left_ear, self.last_right_ear, self.blink_count, 0.0, None
        
        
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        
        left_eye_landmarks = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in left_eye])
        right_eye_landmarks = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in right_eye])
        
        left_ear = self.calculate_ear(left_eye_landmarks)
        right_ear = self.calculate_ear(right_eye_landmarks)

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
    
### 결과 전송 ###
AGGREGATOR_URL = "https://result-aggregator-service/aggregate"

def send_result(final_concentration_score, video_id):
    data = {
        "video_id": video_id,
        "final_concentration_score": final_concentration_score
    }
    response = requests.post(AGGREGATOR_URL, json=data)

    if response.status_code != 200:
        print(f"Failed to send result: {response.text}")

### 세션 관리 ###
sessions = {}

class Session:
    def __init__(self):
        self.blink_detector = BlinkDetector()
        self.total_concentration = 0
        self.frame_count = 0
        self.start_frame = 0
        self.end_frame = None
        self.total_concentration_scores = []

def get_session(session_key):
    if session_key not in sessions:
        sessions[session_key] = Session()
    return sessions[session_key]

### api 엔드포인트 정의 ###
@app.route('/process_frame', methods=['POST'])
def process_frame():
    video_id = request.form['video_id']
    frame_number = int(request.form['frame_number'])
    is_last_frame = request.form['last_frame'].lower() == 'true'

    ip_address = request.remote_addr
    session_key = f"{ip_address}_{video_id}"

    session = get_session[session_key]

    if not is_last_frame:
        frmae_file = request.files['frame']
        frame_data = frmae_file.read()
        frame = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        # 깜빡임 감지
        left_ear, right_ear, blink_count, concentration, face_landmarks = session.blink_detector.detect_blink(frame)
    
        # 집중도 데이터 업데이트
        if face_landmarks is not None:
            avg_ear = (left_ear + right_ear) / 2.0
            session.total_concentration += concentration
            session.frame_count += 1
            session.total_concentration_scores.append(concentration)

        return jsonify({"status": "success", "message": "Frame processed"}), 200

    else:
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
        send_result(final_concentration_score, video_id)
        del sessions[session_key]
        return jsonify({"status": "success", "message": "Video processing completed", "final_concentration_score": final_concentration_score}), 200
    
    return jsonify({"status": "success", "message": "Frame processed"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
