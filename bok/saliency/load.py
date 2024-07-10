import json
import numpy as np
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 환경 변수에서 MongoDB 연결 정보 읽기
username = os.getenv('MONGODB_USERNAME')
password = os.getenv('MONGODB_PASSWORD')
host = os.getenv('MONGODB_HOST')
port = os.getenv('MONGODB_PORT')
database = os.getenv('MONGODB_DB')

# MongoDB 연결 설정
client = MongoClient(f'mongodb://{username}:{password}@{host}:{port}/{database}?authSource=admin')
db = client[database]
fs = gridfs.GridFS(db)

def load_saliency_map(frame_number):
    # GridFS에서 파일 검색
    file = fs.find_one({'frame_number': frame_number})
    if file is None:
        print(f"No saliency map found for frame {frame_number}")
        return None
    
    # JSON 문자열을 읽어서 numpy 배열로 변환
    saliency_map_json = file.read().decode('utf-8')
    saliency_map_list = json.loads(saliency_map_json)
    saliency_map = np.array(saliency_map_list)
    
    return saliency_map

# 특정 프레임의 saliency map을 불러오기 예제
frame_number = 1  # 불러올 프레임 번호
saliency_map = load_saliency_map(frame_number)

print(f'frame number: {frame_number}\n {saliency_map}')