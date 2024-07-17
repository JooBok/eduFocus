##############################################################################################################################
# 큐 제거, 한번 요청한 사항은 한번 밖에 수행 안됨. 다시 말해서 새로 받고 싶으면 다시 프로듀서 부터 시작해서 메시지 전송해야됨.
# 모든 메시지 받음. -> fetch_max_bytes랑 max_partition_fetch_bytes 값 조정 해서 해결.
##############################################################################################################################
from kafka import KafkaConsumer
import requests
import json
import os
import time

# 직접 URL 설정
gaze_url = os.getenv('GAZE_URL')
emotion_url = os.getenv("EMOTION_URL")

def send_data(url, data):
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print(f"Successfully sent frame{data['frame_number']} to {url}")
    else:
        print(f"Failed to send frame {data['frame_number']} to {url}, status code: {response.status_code}")

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

if __name__ == '__main__':
    kafka_topic = 'video-stream'
    kafka_bootstrap_servers = ['localhost:9092']
    save_path = './json_frames'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
