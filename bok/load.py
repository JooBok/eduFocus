import os
import json
import gridfs
from pymongo import MongoClient
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(dotenv_path='/app/.env')

# 환경 변수에서 MongoDB 연결 정보 읽기
username = os.getenv('MONGODB_USERNAME')
password = os.getenv('MONGODB_PASSWORD')
host = os.getenv('MONGODB_HOST')
port = os.getenv('MONGODB_PORT')
database = os.getenv('MONGODB_DB')

# MongoDB 연결 설정
client = MongoClient(f'mongodb://{username}:{password}@{host}:{port}/{database}?authSource=admin')
db = client[database]

def fetch_data_from_collection(collection_name, filename):
    fs = gridfs.GridFS(db, collection=collection_name)
    file_data = fs.find_one({'filename': filename})
    if file_data:
        saliency_map_json = file_data.read()
        saliency_map = json.loads(saliency_map_json)
        return saliency_map
    else:
        print(f"No data found for filename {filename} in collection {collection_name}")
        return None

# 예시: contents1 컬렉션에서 특정 파일의 데이터를 가져오기
collection_name = 'contents1'
filename = 'example_image.jpg'  # 불러오고자 하는 파일 이름

saliency_map = fetch_data_from_collection(collection_name, filename)
if saliency_map:
    print(f"Saliency map for {filename} in collection {collection_name}:")
    print(saliency_map)
