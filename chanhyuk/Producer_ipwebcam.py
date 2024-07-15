from flask import Flask, request, jsonify
from kafka import KafkaProducer
import cv2
import threading
import logging
import json

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
    max_in_flight_requests_per_connection=5,
    request_timeout_ms=60000,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('StreamLogger')

class StreamThread(threading.Thread):
    def __init__(self, video_id, max_frames, ip_address):
        threading.Thread.__init__(self)
        self.stream_url = f'http://{ip_address}:8080/video'
        self.cap = cv2.VideoCapture(self.stream_url)
        self.running = True
        self.video_id = video_id
        self.max_frames = max_frames
        self.ip_address = ip_address

        # 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 프레임 레이트 설정 (예: 30FPS)
        self.cap.set(cv2.CAP_PROP_FPS, 20)

    def run(self):
        if not self.cap.isOpened():
            logger.error("Unable to open video stream.")
            return

        frame_number = 0
        # last_frame_number = 0

        try:
            while self.running and frame_number < self.max_frames:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Unable to read frame. Retrying...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.stream_url)
                    continue

                _, buffer = cv2.imencode('.jpg', frame)
                frame_number += 1

                # if frame_number > self.max_frames:
                #     logger.info("Maximum frame limit reached. Stopping stream.")
                #     self.running = False
                #     break

                # last_frame_number = frame_number

                frame_data = {
                    'video_id': self.video_id,
                    'ip_address': self.ip_address,
                    'frame_number': frame_number,
                    'frame': buffer.tobytes().hex(),
                    'last_frame_state': False
                }

                try:
                    future = producer.send(kafka_topic, frame_data)
                    result = future.get(timeout=10)
                    logger.info(f"Sent frame {frame_number}")
                except Exception as e:
                    logger.error(f"Failed to send frame {frame_number}: {e}")

            # Send end of stream message
            end_message = {
                'video_id': self.video_id,
                'ip_address': self.ip_address,
                'frame_number': frame_number,
                'frame': buffer.tobytes().hex(),
                'last_frame_state': True
            }
            producer.send(kafka_topic, end_message)
            logger.info("Sent end of stream message.")
        except Exception as e:
            logger.error(f"Exception occurred: {e}")

        finally:
            if self.cap.isOpened():
                self.cap.release()

def stop_stream_thread():
    global stream_thread
    if stream_thread and stream_thread.is_alive():
        stream_thread.running = False
        stream_thread.join()

@app.route('/start_stream', methods=['POST'])
def start_stream():
    video_id = request.json.get('video_id', 'default_video_id')
    # duration = request.json.get('duration')  # duration in seconds
    max_frames = request.json.get('max_frames')
    ip_address = request.json.get('ip_address')
    global stream_thread
    stream_thread = StreamThread(video_id, max_frames, ip_address)
    stream_thread.start()
    # Set a timer to stop the stream after the specified duration
    # threading.Timer(duration, stop_stream_thread).start()
    return jsonify({"message": "Stream started"}), 200

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    stop_stream_thread() 
    return jsonify({"message": "Stream stopped"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

