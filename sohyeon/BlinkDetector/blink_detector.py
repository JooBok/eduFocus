import cv2
import mediapipe as mp
import numpy as np
import time

class BlinkDetector:
    def __init__(self, ear_threshold=0.25, consecutive_frames=3, debug_interval=500,
                 min_detection_confidence=0.7, min_presence_confidence=0.7, min_tracking_confidence=0.7):
        # BlinkDetector 클래스 초기화
        # ear_threshold: EAR 임계값. 이 값보다 낮으면 눈이 감긴 것으로 간주
        # consecutive_frames: EAR 값이 임계값보다 낮은 연속된 프레임 수
        # debug_interval: 디버깅 정보를 출력할 프레임 간격

        # MediaPipe FaceMesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,  # 최대 1개의 얼굴 감지
            refine_landmarks=True,  # 더 정밀한 랜드마크 사용
            min_detection_confidence=min_detection_confidence,  # 최소 얼굴 인식 신뢰도
            min_tracking_confidence=min_tracking_confidence  # 최소 얼굴 추적 신뢰도
        )

        # 초기화 변수
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.ear_threshold = ear_threshold  # EAR 임계값
        self.consecutive_frames = consecutive_frames  # 연속 프레임 수
        self.frame_counter = 0  # 현재 연속 프레임 수 카운터
        self.blink_count = 0  # 총 깜빡임 횟수
        self.ear_sequence = []  # EAR 값 시퀀스
        self.sequence_length = 60  # 3초 동안의 데이터 (20 FPS 기준)
        self.start_time = time.time()  # 시작 시간
        self.last_left_ear = None  # 이전 프레임의 왼쪽 EAR 값
        self.last_right_ear = None  # 이전 프레임의 오른쪽 EAR 값
        self.debug_interval = debug_interval  # 디버깅 출력 간격
        self.total_frames = 0  # 총 프레임 수
        self.minute_blinks = 0  # 분당 깜빡임 횟수
        self.minute_concentration = []
        self.max_ear = 0.3  # 초기 최대 EAR 값 (예시)

    def calculate_ear(self, eye_landmarks):
        # EAR (Eye Aspect Ratio) 계산
        # 수직 거리(눈의 위쪽과 아래쪽 점 사이의 거리 (p2-p6와 p3-p5)) 계산
        eye_landmarks = np.array(eye_landmarks)
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        # 수평 거리(눈의 왼쪽과 오른쪽 점 사이의 거리 (p1-p4)) 계산
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        # EAR 계산 (EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|))
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
        self.total_frames += 1  # 처리된 총 프레임 수 증가

        if not results.multi_face_landmarks:
            # 얼굴이 감지되지 않은 경우
            cv2.putText(frame, "Face Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return self.last_left_ear, self.last_right_ear, self.blink_count, 0.0, None
        
        # 얼굴이 감지된 경우
        cv2.putText(frame, "Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 감지된 얼굴 랜드마크
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # 왼쪽 눈과 오른쪽 눈의 랜드마크 인덱스
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        
        # 눈의 랜드마크 좌표 추출
        left_eye_landmarks = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in left_eye])
        right_eye_landmarks = np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in right_eye])
        
        # EAR 계산
        left_ear = self.calculate_ear(left_eye_landmarks)
        right_ear = self.calculate_ear(right_eye_landmarks)

        e_l = self.calculate_blink_value(left_ear)
        e_r = self.calculate_blink_value(right_ear)

        detection_confidence = self.min_detection_confidence
        presence_confidence = self.min_presence_confidence
        tracking_confidence = self.min_tracking_confidence

        c_l = (detection_confidence + presence_confidence + tracking_confidence) / 3
        c_r = (detection_confidence + presence_confidence + tracking_confidence) / 3

        # 디버깅 출력
        if self.total_frames % self.debug_interval == 0:
            print(f"Debug - Left EAR: {left_ear:.4f}, Right EAR: {right_ear:.4f}, e_l: {e_l:.4f}, e_r: {e_r:.4f}, c_l: {c_l:.4f}, c_r: {c_r:.4f}")
        
        # 평균 EAR 계산
        avg_ear = (left_ear + right_ear) / 2.0

        # 눈이 감긴 상태를 감지
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

        if len(self.minute_concentration) >= 60:  # Assuming 20 FPS, this gives us 1 minute of data
            avg_concentration = np.mean(self.minute_concentration)
            self.minute_concentration = []
            self.minute_blinks = 0
            print(f"Minute Concentration: {avg_concentration:.2f}")

        # EAR 값 업데이트
        self.last_left_ear = left_ear
        self.last_right_ear = right_ear

        # 눈의 랜드마크 그리기
        for idx in left_eye:
            x = int(face_landmarks[idx].x * frame.shape[1])
            y = int(face_landmarks[idx].y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        for idx in right_eye:
            x = int(face_landmarks[idx].x * frame.shape[1])
            y = int(face_landmarks[idx].y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

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
        
        return self.calculate_concentration_score_with_time_weight(frame_number)

    def calculate_concentration_score_with_time_weight(self, frame_number):
        # 시간에 따른 가중치를 적용한 집중도 점수 계산
        if not self.ear_sequence:
            return 0
        
        # 현재 시간과 경과 시간 계산
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 가중치 계산
        weights = [self.calculate_time_weight(i, elapsed_time) for i in range(len(self.ear_sequence))]
        weight_sum = sum(weights)
        
        if weight_sum == 0:
            return 0

        # 가중치 적용한 집중도 점수 계산
        concentration_score = sum(ec * w for ec, w in zip(self.ear_sequence, weights)) / weight_sum
        normalized_score = max(0, min(concentration_score, 1.0))  # 집중도 점수를 0과 1 사이로 제한
        
        return normalized_score

    def calculate_time_weight(self, index, elapsed_time):
        # 시간에 따른 가중치 계산
        # index: EAR 시퀀스의 인덱스
        # elapsed_time: 경과 시간
        m = len(self.ear_sequence)
        if m == 0:
            return 0

        t = min(elapsed_time, 30)  # 최대 30초까지만 고려
        weight = max(0.01, min(1, (m - index) / m * (30 - t) / 30))  # 최소 가중치 값 설정
        
        return weight
