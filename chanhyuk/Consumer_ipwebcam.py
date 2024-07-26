from kafka import KafkaConsumer
import requests
import json
import os
import time

# 직접 URL 설정
gaze_url = os.getenv('GAZE_URL')
emotion_url = os.getenv("EMOTION_URL")

# gaze 모델과 emotion 모델의 url로 스트림한 데이터를 보냄.
def send_data(url, data):
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print(f"Successfully sent frame{data['frame_number']} to {url}")
    else:
        print(f"Failed to send frame {data['frame_number']} to {url}, status code: {response.status_code}")

# 컨슈머에서 프로듀서에서 전송받은 데이터를 polling을 통해 받아서
save_path경로에 json형태로 저장. 
def consume_and_save_frames(consumer, save_path, gaze_url, emotion_url):
    while True:
        print('Polling for messages...')
        message_pack = consumer.poll(timeout_ms = 1000)
        if not message_pack:
            print("No message available. Waiting for message...")
            time.sleep(0.1)
            continue
        for tp, messages in message_pack.items():
            for message in messages:
                frame_data = json.loads(message.value)
                # latest_frame_date = frame_data
                print(f"Received frame {frame_data['frame_number']} for video_id {frame_data['video_id']}")

                #gaze와 emotion URL로 동기 요청 보내기
                send_data(gaze_url, frame_data)
                send_data(emotion_url, frame_data)
                
                
                video_id = frame_data['video_id']
                frame_number = frame_data['frame_number']
                json_filename = os.path.join(save_path, f"{video_id}_frame_{frame_number}.json")
                with open(json_filename, 'w') as json_file:
                    json.dump(frame_data, json_file)
                print(f"Saved latest frame {frame_number} to {json_filename}")

                last_frame_state = frame_data.get('last_frame_state', False)
                if last_frame_state:
                    print(f"Stream ended for video_id {video_id} after {frame_number} frames.")
                    return
# 9092 포트로 데이터 
if __name__ == '__main__':
    kafka_topic = 'video-stream'
    kafka_bootstrap_servers = ['localhost:9092']
    save_path = './json_frames'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # kafka 컨슈머 설정. 
    consumer = KafkaConsumer(
        kafka_topic,
        bootstrap_servers=kafka_bootstrap_servers,
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='video-stream-group',
        value_deserializer=lambda x: x.decode('utf-8'),
        fetch_max_bytes=1024 * 1024,  # 1MB
        max_partition_fetch_bytes=1024 * 1024  # 1MB
    )

    print("Listening for messages on 'video-stream' topic...")
    
    try:
        consume_and_save_frames(consumer, save_path)
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        consumer.close()

    print("Program terminated gracefully.")
