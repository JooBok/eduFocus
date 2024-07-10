from flask import Flask, request, jsonify
from kafka import KafkaProducer
import cv2
import threading
import logging


app = Flask(__name__)

# Kafka 설정
kafka_topic = 'video-stream'
kafka_bootstrap_servers = ['localhost:9092']

#디버깅을 위한 로깅 설정
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger('StreamLogger')
# kafka_logger = logging.getLogger('kafka')


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

        # 프레임 레이트 설정 (예: 30FPS)
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

                # # 프레임 스킵 (예: 2번째 프레임마다 전송)
                # frame_skip_count += 1
                # if frame_skip_count % 2 != 0:
                #     continue

                _, buffer = cv2.imencode('.jpg', frame)
                try:
                    future = producer.send(kafka_topic, buffer.tobytes())
                    result = future.get(timeout=15)  # 전송 완료를 기다림
                    logger.info(f"Sent frame with result: {result}")
                    producer.flush()  # 데이터 전송을 확실히 함
                except Exception as e:
                    logger.error(f"Failed to send frame : {e}")

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

# stream_thread = StreamThread()
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
    # topic = request.json.get('topic')
    # if not topic:
    #     return jsonify({'error' : 'Topic name is required'}), 400
    # if not stream_thread or not stream_thread.is_alive():
    #     stream_thread.stop()
    #     return jsonify({'status' : 'Streaming stopped'}), 200
    # else:
    #     return jsonify({'status' : 'No streaming to stop'}), 200
    
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
# def start_stream():
#     global stream_thread
#     if not stream_thread.is_alive():
#         stream_thread = StreamThread()
#         stream_thread.start()
#         return jsonify({'status': 'Streaming started'}), 200
#     else:
#         return jsonify({'status': 'Streaming already running'}), 200

# @app.route('/stop_stream', methods=['POST'])
# def stop_stream():
#     global stream_thread
#     if stream_thread.is_alive():
#         stream_thread.stop()
#         return jsonify({'status': 'Streaming stopped'}), 200
#     else:
#         return jsonify({'status': 'No streaming to stop'}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
################### 바로 위에 코드 사용해서 제대로 전송 되는거 확인 문제는 프레임마다 topic을 생성해서 전송한다는 문제 있음
################### 6/31일에 와서 하나의 topic에 모두 전송되도록 수정하고 컨슈머 수정하기 아마 코드 다시 수정해야 할듯 upper code라고 되어 있는 부분이 원래 하나의 topic에 프레임 저장할 수 있도록 하는 코드였음.
################### 컨슈머도 알맞게 수정.(png 파일로 변환 하거나 관련 요구 사항 수용해서 변형)

################## 원인은 server.properties에서 PLAINTEXT://0.0.0.0:9092로 고쳤더니 됐었음.......... 정확한 이유는 아닐수도 있지만 유력 원인.
################## 7/1 단일 topic에 이미지 전송 확인. 컨슈머로 이미지 로딩 확인 -> 로컬 폴더에 이미지 저장함.
################## 배포를 위한 버전 정보 저장과 코드 통일 kafka-python==1.4.3, api version==(0,2,0)