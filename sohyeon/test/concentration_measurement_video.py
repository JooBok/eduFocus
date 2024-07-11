import cv2
import time
import csv
import numpy as np
from datetime import datetime
import os
from blink_detector import BlinkDetector

class ConcentrationMeasurement:
    def __init__(self, video_path, content_duration=180):
        self.video_path = video_path
        self.content_duration = content_duration
        self.save_path = "C:\\Users\\BIG03-01\\Documents\\kdt7\\eduu\\concentration_data"
        os.makedirs(self.save_path, exist_ok=True)

    def measure(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blink_detector = BlinkDetector()
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_FPS, 20) # 초당 20프레임
        
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return
        
        total_concentration = 0
        frame_count = 0
        start_time = time.time()
        
        

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            left_ear, right_ear, blink_count, concentration, face_landmarks = blink_detector.detect_blink(frame)
            
            if left_ear is not None and right_ear is not None:
                avg_ear = (left_ear + right_ear) / 2.0
                total_concentration += concentration
                frame_count += 1
                print(f"Frame {frame_count}: Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, Concentration: {concentration:.2f}")
                
                cv2.putText(frame, f"Avg EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Blinks: {blink_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Concentration: {concentration:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                print("Failed to detect face landmarks")
                        
            elapsed_time = time.time() - start_time
            if elapsed_time > self.content_duration:
                break
            
        
        cap.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            average_concentration = total_concentration / frame_count
            concentration_percentage = average_concentration * 100
        else:
            concentration_percentage = 0
        
        # normalized_concentration 변수를 정의합니다.
        normalized_concentration = concentration_percentage  # 또는 적절한 값으로 정의

        # 시간 가중치 적용 및 최종 집중 점수 계산
        total_minutes = self.content_duration / 60
        weightings = np.linspace(1, 0.5, int(total_minutes))
        concentration_scores = [normalized_concentration * w for w in weightings]
        final_concentration_score = np.mean(concentration_scores)

        print(f"이번 콘텐츠의 집중도는 {final_concentration_score:.2f}%였습니다.")
        print(f"Total frames processed: {frame_count}")

        # 결과를 CSV 파일에 추가로 기록
        result_csv_file = os.path.join(self.save_path, f'final_result_{timestamp}.csv')
        with open(result_csv_file, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Final Concentration Score', 'Total Frames'])
            csv_writer.writerow([final_concentration_score, frame_count])

if __name__ == "__main__":
    video_path = "C:\\Users\\BIG03-01\\Documents\\kdt7\\eduu\\concentration_data\\test1.mp4"
    cm = ConcentrationMeasurement(video_path, 180)
    cm.measure()
