# import cv2
# from kafka import KafkaProducer

# def stream_to_kafka():
#     stream_url = 'http://10.41.0.99:8080/video'

#     kafka_topic = 'video-stream'
#     kafka_bootstrap_servers = ['localhost:9092']
#     producer = KafkaProducer(bootstrap_servers=kafka_bootstrap_servers)

#     cap= cv2.VideoCapture(stream_url)

#     if not cap.isOpened():
#         print("Error: Unable to open video stream.")
#         return
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Unable to read frame.")
#                 break
            
#             producer.send(kafka_topic, frame.tobytes())

#             cv2.imshow('IP Webcam Stream', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     except KeyboardInterrupt:
#         print("Streaming stopped by user.")
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         producer.close()

# if __name__ == "__main__":
#     stream_to_kafka()
###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################
# from flask import Flask, request, jsonify
# from kafka import KafkaProducer
# import cv2
# import threading
# import logging

# app = Flask(__name__)

# # Kafka 설정
# kafka_topic = 'video-stream'
# kafka_bootstrap_servers = ['localhost:9092']
# producer = KafkaProducer(bootstrap_servers=kafka_bootstrap_servers,linger_ms=10,batch_size=16384)

# #ip webcam 스트림 url
# stream_url = 'http://10.41.0.99:8080/video'

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger('StreamLogger')

# class StreamThread(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#         self.cap = cv2.VideoCapture(stream_url)
#         self.running = True

#         # 해상도 설정
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#         # 프레임 레이트 설정 (예: 15FPS)
#         self.cap.set(cv2.CAP_PROP_FPS, 15)

#         #FFmpeg 설정
#         self.ffmpeg_process = None

#     def run(self):
#         if not self.cap.isOpened():
#             print("Error : Unable to open video stream.")
#             return
#         frame_skip_count=0
#         try:
#             while self.running:
#                 ret, frame = self.cap.read()
#                 if not ret:
#                    logger.error("Error: Unable to read frame. Retrying...")
#                    self.cap.release()
#                    self.cap = cv2.VideoCapture(stream_url)
#                    continue
#                 # 프레임 스킵(예: 2번째 프레임마다 전송)
#                 frame_skip_count +=1
#                 if frame_skip_count % 2 != 0:
#                     continue
#                 _,buffer = cv2.imencode('.jpg', frame)
#                 producer.send(kafka_topic, buffer.tobytes())
#                 logger.info(f"Sent frame {frame_skip_count // 2}")

#                 cv2.imshow('IP Webcam Stream', frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#         except KeyboardInterrupt:
#             print("Streaming stopped by user.")
#         finally:
#             self.cap.release()
#             cv2.destroyAllWindows()
#     def stop(self):
#         self.running = False
# stream_thread = StreamThread()

# @app.route('/start_stream', methods=['POST'])
# def start_stream():
#     global stream_thread
#     if not stream_thread.is_alive():
#         stream_thread = StreamThread()
#         stream_thread.start()
#         return jsonify({'status' : 'Streaming started'}), 200
#     else:
#         return jsonify({'status' : 'Streaming already running'}), 200
#     # cap = cv2.VideoCapture(stream_url)
#     # if not cap.isOpened():
#     #     print("Error : Unable to open video stream")
#     #     return
#     # try:
#     #     while True:
#     #         ret, frame = cap.read()
#     #         if not ret:
#     #             print("Error : Unable to read frame")
#     #             break

#     #         #kafka로 전송(원본 프레임 데이터를 메시지로 보냄)
#     #         _, buffer = cv2.imencode('.jpg', frame)
#     #         producer.send(kafka_topic, buffer.tobytes())
#     # except KeyboardInterrupt:
#     #     print("Streaming stopped by user.")
#     # finally:
#     #     cap.release()

# @app.route('/stop_stream', methods=['POST'])
# def stop_stream():
#     global stream_thread
#     if stream_thread.is_alive():
#         stream_thread.stop()
#         return jsonify({'status' : 'Streaming stopped'}), 200
#     else:
#         return jsonify({'status' : 'No streaming to stop'}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=5000)
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
from flask import Flask, request, jsonify
from kafka import KafkaProducer
import cv2
import threading
import logging

app = Flask(__name__)

# Kafka 설정
kafka_topic = 'video-stream'
kafka_bootstrap_servers = ['localhost:9092']
producer = KafkaProducer(
    bootstrap_servers=kafka_bootstrap_servers,
    api_version=(0, 2, 0),
      linger_ms=10,
        batch_size=16384,
          acks='all', 
          retries=5,
            max_in_flight_requests_per_connection=5, request_timeout_ms=60000)

# ip webcam 스트림 url
stream_url = 'http://10.41.0.99:8080/video'

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('StreamLogger')

class StreamThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(stream_url)
        self.running = True

        # 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 프레임 레이트 설정 (예: 15FPS)
        self.cap.set(cv2.CAP_PROP_FPS, 15)

    def run(self):
        if not self.cap.isOpened():
            logger.error("Unable to open video stream.")
            return

        frame_skip_count = 0

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Unable to read frame. Retrying...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(stream_url)
                    continue

                # 프레임 스킵 (예: 2번째 프레임마다 전송)
                frame_skip_count += 1
                if frame_skip_count % 2 != 0:
                    continue

                _, buffer = cv2.imencode('.jpg', frame)
                try:
                    future = producer.send(kafka_topic, buffer.tobytes())
                    result = future.get(timeout=60)  # 전송 완료를 기다림
                    logger.info(f"Sent frame {frame_skip_count // 2} with result: {result}")
                    producer.flush()  # 데이터 전송을 확실히 함
                except Exception as e:
                    logger.error(f"Failed to send frame {frame_skip_count // 2} : {e}")

                cv2.imshow('IP Webcam Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def stop(self):
        self.running = False

stream_thread = StreamThread()

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global stream_thread
    if not stream_thread.is_alive():
        stream_thread = StreamThread()
        stream_thread.start()
        return jsonify({'status': 'Streaming started'}), 200
    else:
        return jsonify({'status': 'Streaming already running'}), 200

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global stream_thread
    if stream_thread.is_alive():
        stream_thread.stop()
        return jsonify({'status': 'Streaming stopped'}), 200
    else:
        return jsonify({'status': 'No streaming to stop'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
