from flask import Flask, request, jsonify
from kafka import KafkaProducer
import cv2
import threading
import logging

app = Flask(__name__)

kafka_topic = 'video-stream'
kafka_bootstrap_servers = ['kafka-service:9092']

stream_url = 'http://10.41.0.99:8080/video'

producer = KafkaProducer(
    bootstrap_servers=kafka_bootstrap_servers,
    api_version=(0, 2, 0),
    linger_ms=10,
    batch_size=16384,
    acks='all', 
    retries=5,
    max_in_flight_requests_per_connection=5,
    request_timeout_ms=60000
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('StreamLogger')


class StreamThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(stream_url)
        self.running = True

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.cap.set(cv2.CAP_PROP_FPS, 30)

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

                _, buffer = cv2.imencode('.jpg', frame)
                try:
                    future = producer.send(kafka_topic, buffer.tobytes())
                    result = future.get(timeout=15)
                    logger.info(f"Sent frame with result: {result}")
                    producer.flush()
                except Exception as e:
                    logger.error(f"Failed to send frame : {e}")
                    
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def stop(self):
        self.running = False

stream_thread = None

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global stream_thread
    if not stream_thread or not stream_thread.is_alive():
        stream_thread = StreamThread()
        stream_thread.start()
        return jsonify({'status' : 'Streaming started'}), 200
    else:
        return jsonify({'status' : 'Streaming already running'}), 200
    
@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global stream_thread
    if stream_thread and stream_thread.is_alive():
        stream_thread.stop()
        stream_thread.join()
        return jsonify({'status' : 'streaming stopped'}), 200
    else:
        return jsonify({'status' : 'No streaming to stop'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)