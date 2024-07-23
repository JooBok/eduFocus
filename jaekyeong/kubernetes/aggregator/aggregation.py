from flask import Flask, request, jsonify
from minio import Minio
import pandas as pd
import json
import requests
from datetime import datetime
import os
from io import BytesIO
import logging
import redis

app = Flask(__name__)
redis_client = redis.Redis(host='redis-service', port=6379, db=3)
# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### minio client ###
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "minio-service:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
)

### classification model URI ###
CLASSIFICATION_MODEL_URL = "http://classification-model-service/predict"

def mk_bucket(bucket_name):
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logger.info(f"Created bucket '{bucket_name}'")
        else:
            logger.info(f"Bucket '{bucket_name}' already exists")
    except Exception as e:
        logger.error(f"Error checking/creating bucket '{bucket_name}': {str(e)}")
        raise

### 데이터를 저장 딕셔너리 ###
data_store = {}
mk_bucket("aggregated-data")

@app.route('/aggregate', methods=['POST'])
def aggregate():
    logger.info("Received request to /aggregate")
    data = request.json
    logger.info(f"Received data: {data}")
    
    video_id = data['video_id']
    model_type = data['model_type']
    ip_address = data['ip_address']
    final_score = data['final_score']

    ### 서로 다른 pod에서 처리되는것을 방지하기 위한 redis사용 ###
    redis_key = f"{ip_address}:{video_id}"
    redis_client.hset(redis_key, model_type, final_score)

    # 현재 저장된 데이터 상태 로깅
    current_data = redis_client.hgetall(redis_key)
    logger.info(f"Current data for {redis_key}: {current_data}")

    ### 3개의 결과가 모두 모였는지 체크 후 진행 ###
    if len(current_data) == 3:
        logger.info(f"All three model types received for video_id: {video_id}")

        processed_data = {k.decode(): float(v.decode()) for k, v in current_data.items()}
        df = pd.DataFrame([processed_data])
        df['ip_address'] = ip_address
        df['video_id'] = video_id
        df['timestamp'] = datetime.now().isoformat()

        ### MinIO에 저장 ###
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)

        file_path = f"{datetime.now().strftime('%Y/%m/%d')}/{ip_address}/{video_id}/"
        file_name = f"{video_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

        try:
            minio_client.put_object(
                "aggregated-data",
                f"{file_path}{file_name}",
                csv_buffer,
                length=csv_buffer.getbuffer().nbytes,
                content_type="text/csv"
            )
            logger.info(f"Successfully uploaded {file_path}{file_name} to MinIO")
        except Exception as e:
            logger.error(f"Failed to upload to MinIO: {str(e)}")
            return jsonify({"status": "error", "message": f"Failed to upload to MinIO: {str(e)}"}), 500

        ### Local에 저장 ###
        try:
            local_path = '/tmp/storage'
            os.makedirs(local_path, exist_ok=True)
            local_file = os.path.join(local_path, "aggregated_data.csv")
            df.to_csv(local_file, index=False, mode='a', header=not os.path.exists(local_file))
            logger.info(f"Successfully saved data to local file: {local_file}")
        except Exception as e:
            logger.error(f"Failed to save data locally: {str(e)}")
            return jsonify({"status": "error", "message": f"Failed to save data locally: {str(e)}"}), 500

        ### Classification model 데이터 전송 (주석 처리됨) ###
        # try:
        #     response = requests.post(CLASSIFICATION_MODEL_URL, json=data_store[ip_address][video_id])
        #     logger.info(f"Successfully sent data to classification model. Response: {response.text}")
        # except Exception as e:
        #     logger.error(f"Failed to send data to classification model: {str(e)}")
        #     return jsonify({"status": "error", "message": f"Failed to send data to classification model: {str(e)}"}), 500
        
        ### 데이터 처리 완료 후 저장소에서 제거 ###
        redis_client.delete(redis_key)
        
        return jsonify({"status": "success", "message": "Data aggregated and stored in MinIO"}), 200
    
    return jsonify({"status": "success", "message": "Data received"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)