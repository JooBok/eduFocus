import cv2
import numpy as np
import time
import csv
from datetime import datetime
import os
import tkinter as tk
from threading import Thread
from blink_detector import BlinkDetector

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
        save_path = "C:\\Users\\BIG03-01\\Documents\\kdt7\\eduu\\concentration_data"
        os.makedirs(save_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blink_detector = BlinkDetector()  # 실시간 집중도 측정
        cap = cv2.VideoCapture(2)  # 카메라 디바이스 번호 (필요에 따라 조정)
        cap.set(cv2.CAP_PROP_FPS, 20)  # FPS 설정
        
        
        if not cap.isOpened():
            print("Error: Could not open video device.")
            return
        
        total_concentration = 0 # 총 집중도 합
        frame_count = 0 # 처리된 프레임 수
        start_time = time.time() # 시작 시간 기록

        # 실시간 영상 처리
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # 눈 깜빡임 및 집중도 감지
            left_ear, right_ear, blink_count, concentration, face_landmarks = blink_detector.detect_blink(frame)
            
            if face_landmarks is not None:
                # 얼굴이 감지된 경우의 처리
                avg_ear = (left_ear + right_ear) / 2.0
                total_concentration += concentration
                frame_count += 1

                current_time = time.time()
                elapsed_time = current_time - start_time


                if frame_count % 50 == 0:
                    print(f"Frame {frame_count}: Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, Concentration: {concentration:.2f}")
                
                # 화면에 정보 표시
                cv2.putText(frame, f"Avg EAR: {avg_ear:.2f}", (10, 70) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Blinks: {blink_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Concentration: {concentration:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                # 얼굴이 감지되지 않은 경우의 처리
                cv2.putText(frame, "No face detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Concentration Measurement', frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
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

        # 결과를 CSV 파일에 추가로 기록 (선택 사항)
        result_csv_file = os.path.join(save_path, f'final_result_{timestamp}.csv')
        with open(result_csv_file, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Final Concentration Score', 'Total Frames'])
            csv_writer.writerow([final_concentration_score, frame_count])


# 몇 분 동안 측정하는 거 말고 ->시작 종료 버튼에 따라 집중도 측정하기(태블릿 뒤에 백그라운드에서 돌릴 거니까)
if __name__ == "__main__":
        root = tk.Tk()
        app = ConcentrationApp(root)
        root.mainloop()