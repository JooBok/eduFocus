from kafka import KafkaConsumer
import requests
import mediapipe as mp

consumer = KafkaConsumer('kid_face_video', bootstrap_servers=['kafka-service:9092'])
aggregator_url = "http://result-aggregator-service/aggregate"

for message in consumer:
    video_data = message.value
    # MediaPipe 처리 로직
    result = process_with_mediapipe(video_data)
    requests.post(aggregator_url, json=result)
