from flask import Flask, request
from kafka import KafkaProducer

app = Flask(__name__)
kafka_server = 'kafka-service:9092'
kafka_topic = 'kid_face_video'
producer = KafkaProducer(bootstrap_servers=[kafka_server])

@app.route('/process_video', methods=['POST'])
def process_video():
    video_data = request.files['video'].read()
    producer.send(kafka_topic, value=video_data)
    return "Video sent for processing", 202

if __name__ == '__main__':
    port = 5000
    app.run(host='0.0.0.0', port=port)
