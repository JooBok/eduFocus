import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import os
import tkinter as tk
from threading import Thread

class BlinkDetector:
    def __init__(self, ear_threshold=0.25, consecutive_frames=3, debug_interval=500,
                 min_detection_confidence=0.7, min_presence_confidence=0.7, min_tracking_confidence=0.7):
        # BlinkDetector 클래스 초기화
        # ear_threshold: EAR 임계값. 이 값보다 낮으면 눈이 감긴 것으로 간주
        # consecutive_frames: EAR 값이 임계값보다 낮은 연속된 프레임 수
        # debug_interval: 디버깅 정보를 출력할 프레임 간격

        # MediaPipe FaceMesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, # 최대 1개의 얼굴 감지
            refine_landmarks=True, # 더 정밀한 랜드마크 사용
            min_detection_confidence=min_detection_confidence, # 최소 얼굴 인식 신뢰도
            min_tracking_confidence=min_tracking_confidence # 최소 얼굴 추적 신뢰도
        )

        # 초기화 변수
        self.ear_threshold = ear_threshold # EAR 임계값
        self.consecutive_frames = consecutive_frames # 연속 프레임 수
        self.frame_counter = 0 # 현재 연속 프레임 수 카운터
        self.blink_count = 0 # 총 깜빡임 횟수
        self.ear_sequence = [] # EAR 값 시퀀스
        self.sequence_length = 60  # 3초 동안의 데이터 (20 FPS 기준)
        self.start_time = time.time() # 시작 시간
        self.last_left_ear = None # 이전 프레임의 왼쪽 EAR 값
        self.last_right_ear = None # 이전 프레임의 오른쪽 EAR 값
        self.debug_interval = debug_interval  # 디버깅 출력 간격q
        self.total_frames = 0 # 총 프레임 수
        self.minute_blinks = 0  # 분당 깜빡임 횟수
        self.minute_concentration = []  
        self.max_ear = 0.3  # 초기 최대 EAR 값 (예시)
        self.ear_history = []

        # 추가 변수 설정
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

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
        blink_value = max(0, self.max_ear - ear)  # 반전된 EAR 값 (=눈 감은 값)
        return blink_value
    
    def update_max_ear(self, ear):
        self.ear_history.append(ear)
        if len(self.ear_history) > 100:  # 최근 100개 샘플만 유지
            self.ear_history.pop(0)
        self.max_ear = np.percentile(self.ear_history, 95)  # 95 백분위수 사용

    def detect_blink(self, frame):
        # 프레임에서 얼굴의 눈 깜빡임을 감지
        results = self.mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.total_frames += 1 # 처리된 총 프레임 수 증가
        last_process_time = time.time() # 마지막 처리 시간 업데이트 위치 수정

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

        self.update_max_ear(left_ear)
        self.update_max_ear(right_ear)

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
        concentration = self.calculate_concentration(e_l, e_r, c_l, c_r)
        self.minute_concentration.append(concentration)

        if len(self.minute_concentration) >= 60:  # Assuming 20 FPS, this gives us 1 minute of data
            avg_concentration = np.mean(self.minute_concentration)
            self.minute_concentration = []
            self.minute_blinks = 0
            print(f"Minute Concentration: {avg_concentration:.2f}")
            # Save minute concentration data to CSV or process further

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

    def calculate_concentration(self, e_l, e_r, c_l, c_r):
        e_l_normalized = max(0, min(e_l, 1))
        e_r_normalized = max(0, min(e_r, 1))

        if (c_l + c_r) > 0:
            e_c = (c_l * e_l_normalized + c_r * e_r_normalized) / (c_l + c_r)
        else:
            e_c = (e_l_normalized + e_r_normalized) / 2

        print(f"e_l: {e_l}, e_r: {e_r}, c_l: {c_l}, c_r: {c_r}, e_c: {e_c}")

        if e_c == 0:
            concentration = 0
        else:
            concentration = 1 / e_c

        print(f"Calculated concentration: {concentration}")
        
        self.ear_sequence.append(concentration)
        if len(self.ear_sequence) > self.sequence_length:
            self.ear_sequence.pop(0)
        
        return self.calculate_concentration_score()

    def calculate_concentration_score(self):
        # 시간에 따른 가중치를 적용한 집중도 점수 계산
        if not self.ear_sequence:
            return 0
        
        # 현재 시간과 경과 시간 계산
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 가중치 계산
        weights = [self.calculate_weight(i, elapsed_time) for i in range(len(self.ear_sequence))]
        weight_sum = sum(weights)
        if weight_sum == 0:
            return 0

        # 가중치 적용한 집중도 점수 계산
        concentration_score = sum(ec * w for ec, w in zip(self.ear_sequence, weights)) / weight_sum
        
        # 집중도 점수를 0과 1 사이로 제한
        return max(0, min(concentration_score, 1.0))  # 0과 1 사이의 값으로 제한

    def calculate_weight(self, index, elapsed_time):
        # 시간에 따른 가중치 계산
        # index: EAR 시퀀스의 인덱스
        # elapsed_time: 경과 시간
        m = len(self.ear_sequence)
        t = min(elapsed_time, 30)  # 최대 30초까지만 고려
        return max(0, min(1, (m - index) / m * (30 - t) / 30))

class ConcentrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Concentration Measurement")
        self.is_running = False
        self.thread = None

        self.start_button = tk.Button(root, text="Start", command=self.start_measurement)
        self.start_button.pack(pady=10)
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_measurement)
        self.stop_button.pack(pady=10)

    def start_measurement(self):
        if not self.is_running:
            self.is_running = True
            self.thread = Thread(target=self.real_time_concentration_measurement)
            self.thread.start()

    def stop_measurement(self):
        if self.is_running:
            self.is_running = False
            if self.thread:
                self.thread.join()

    def real_time_concentration_measurement(self):
        # 지정된 경로 설정
        save_path = "C:\\Users\\BIG03-01\\Documents\\kdt7\\edufocus\\concentration_data"
        os.makedirs(save_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file=os.path.join(save_path, f'concentration_data_{timestamp}.csv')
        detailed_csv_file=os.path.join(save_path, f'detailed_landmarks_{timestamp}.csv')

        # 실시간 집중도 측정
        blink_detector = BlinkDetector()
        cap = cv2.VideoCapture(2)  # 카메라 디바이스 번호 (필요에 따라 조정)
        cap.set(cv2.CAP_PROP_FPS, 20)  # FPS 설정
        
        if not cap.isOpened():
            print("Error: Could not open video device.")
            return
        
        total_concentration = 0 # 총 집중도 합
        frame_count = 0 # 처리된 프레임 수
        start_time = time.time() # 시작 시간 기록

        process_interval = 1
        last_process_time = time.time()
        
        # CSV 파일 생성 및 초기화
        with open(csv_file, mode='w', newline='') as file, open(detailed_csv_file, mode='w', newline='') as detailed_file:
            writer = csv.writer(file)
            detailed_writer = csv.writer(detailed_file)
            writer.writerow(['Frame', 'Left EAR', 'Right EAR', 'Blinks', 'Concentration'])
            detailed_writer.writerow(['Frame', 'Landmark', 'x', 'y'])
            
            last_save_time = time.time()  # 마지막 저장 시간 초기화

            # 실시간 영상 처리
            while cap.isOpened() and self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break

                current_time = time.time()
                if current_time - last_process_time >= process_interval:
                    last_process_time = current_time # 마지막 처리 시간 업데이트
                    # 눈 깜빡임 및 집중도 감지
                    left_ear, right_ear, blink_count, concentration, face_landmarks = blink_detector.detect_blink(frame)
                
                    if face_landmarks is not None:
                    # 얼굴이 감지된 경우의 처리
                        avg_ear = (left_ear + right_ear) / 2.0
                        total_concentration += concentration
                        frame_count += 1

                        elapsed_time = current_time - start_time

                        print(f"Frame {frame_count}: Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, Concentration: {concentration:.2f}")
                    

                        # CSV 파일에 데이터 기록 (10초마다 데이터 저장)
                        if current_time - last_save_time >= 10:
                            writer.writerow([f"{elapsed_time:.2f}", left_ear, right_ear, blink_count, concentration])
                            for i, landmark in enumerate(face_landmarks):
                                detailed_writer.writerow([f"{elapsed_time:.2f}", i, landmark.x, landmark.y])
                            last_save_time = current_time  # 마지막 저장 시간 업데이트
                    
                        # 화면에 정보 표시
                        cv2.putText(frame, f"Avg EAR: {avg_ear:.2f}", (10, 70) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Blinks: {blink_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Concentration: {concentration:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # detailed landmarks data 기록 (x, y)
                        for i, landmark in enumerate(face_landmarks):
                            detailed_writer.writerow([frame_count, i, landmark.x, landmark.y])
                    else:
                        # 얼굴이 감지되지 않은 경우의 처리
                        cv2.putText(frame, "No face detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    last_process_time = current_time # 마지막 처리 시간 업데이트

                    cv2.imshow('Concentration Measurement', frame)
                
                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        cap.release()
        cv2.destroyAllWindows()
        
        # 최종 집중도 계산 및 출력
        if frame_count > 0:
            average_concentration = total_concentration / frame_count
            concentration_percentage = average_concentration * 100
        else:
            concentration_percentage = 0

        # normalized_concentration 변수 정의
        normalized_concentration = concentration_percentage  # 또는 적절한 값으로 정의

        # 시간 가중치 적용 및 최종 집중 점수 계산
        total_minutes = (time.time() - start_time) / 60
        weightings = np.linspace(1, 0.5, int(total_minutes))
        concentration_scores = [normalized_concentration * w for w in weightings]
        final_concentration_score = np.mean(concentration_scores)

        print(f"이번 콘텐츠의 집중도는 {final_concentration_score:.2f}%였습니다.")
        print(f"Total frames processed: {frame_count}")

        # 최종 결과를 CSV 파일에 추가로 기록 
        result_csv_file = os.path.join(save_path, f'final_result_{timestamp}.csv')
        with open(result_csv_file, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Final Concentration Score', 'Total Frames'])
            csv_writer.writerow([final_concentration_score, frame_count])

        # 그래프 생성
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        time_stamps = data[:, 0]
        concentrations = data[:, 4]

        plt.figure(figsize=(10, 6))
        plt.plot(time_stamps, concentrations, marker='o', linestyle='-', color='b')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Concentration')
        plt.title('Concentration Over Time')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f'concentration_over_time_{timestamp}.png'))
        plt.show()

# 몇 분 동안 측정하는 거 말고 ->시작 종료 버튼에 따라 집중도 측정하기(태블릿 뒤에 백그라운드에서 돌릴 거니까)
if __name__ == "__main__":
        root = tk.Tk()
        app = ConcentrationApp(root)
        root.mainloop()