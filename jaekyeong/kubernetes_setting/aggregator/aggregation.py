from flask import Flask, request, jsonify
from minio import Minio
import pandas as pd
import json
import requests
from datetime import datetime
import os

app = Flask(__name__)

### minio client ###
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "minio-service:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
)

### classification model URI ###
CLASSIFICATION_MODEL_URL = "http://classification-model-service/predict"

### 데이터를 저장 딕셔너리 ###
data_store = {}

@app.route('/aggregate', methods=['POST'])
def aggregate():
    data = request.json
    video_id = data['video_id']
    model_type = data['model_type']
    ip_address = data['ip_address']
    final_score = data['final_score']

    if ip_address not in data_store:
        data_store[ip_address] = {}
    if video_id not in data_store[ip_address]:
        data_store[ip_address][video_id] = {}
    
    data_store[ip_address][video_id][model_type] = final_score
    
    ### 3개의 결과가 모두 모였는지 체크 후 진행 ###
    if len(data_store[ip_address][video_id]) == 3:
        df = pd.DataFrame([data_store[ip_address][video_id]])
        df['ip_address'] = ip_address
        df['video_id'] = video_id
        df['timestamp'] = datetime.now().isoformat()

        ### MinIO에 저장 ###
        csv_buffer = df.to_csv(index=False).encode('utf-8')
        
        minio_client.put_object(
            "aggregated-data",              # bucket
            f"{video_id}.csv",              # file name
            csv_buffer,                     # data
            len(csv_buffer),                # Content length
            content_type="application/csv"  # type 설정정
        )

        ### Local에 저장 ###
        local_path = '/home/fp/storage'
        os.makedirs(local_path, exist_ok=True)
        local_file = os.path.join(local_path, "aggregated_data.csv")
        ### 없으면 헤더 있게 생성, 있으면 append ###
        if not os.path.exists(local_file):
            df.to_csv(local_file, index=False, mode='w')
        else:
            df.to_csv(local_file, index=False, mode='a', header=False)

        ### Classification model 데이터 전송 ###
        ### 전송 데이터 예시 {"emotion": 0.75, "gaze": 0.82, "blink": 0.63} ###     
        response = requests.post(CLASSIFICATION_MODEL_URL, json=data_store[ip_address][video_id])
        
        ### 데이터 처리 완료 후 저장소에서 제거 ###
        del data_store[ip_address][video_id]

        return jsonify({"status": "success", "message": "Data aggregated and sent to classification model"}), 200
    
    return jsonify({"status": "success", "message": "Data received"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
