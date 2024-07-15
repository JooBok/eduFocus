import os
import numpy as np
import gridfs
from dotenv import load_dotenv
from pymongo import MongoClient, errors

def load_from_gridfs(db, collection_name, filename):
    fs = gridfs.GridFS(db, collection=collection_name)
    file_data = fs.find_one({'filename': filename})
    if file_data:
        return file_data.read(), file_data.metadata['img_size']
    else:
        print(f"File {filename} not found in GridFS collection {collection_name}")
        return None, None

if __name__ == '__main__':
    # .env 파일 로드
    load_dotenv(dotenv_path='/app/.env')

    # 환경 변수에서 MongoDB 연결 정보 읽기
    username = os.getenv('MONGODB_USERNAME')
    password = os.getenv('MONGODB_PASSWORD')
    host = os.getenv('MONGODB_HOST')
    port = os.getenv('MONGODB_PORT')
    database = os.getenv('MONGODB_DB')

    # MongoDB 연결 설정
    try:
        client = MongoClient(f'mongodb://{username}:{password}@{host}:{port}/{database}?authSource=admin')
        db = client[database]
        print("Successfully connected to MongoDB")
    except errors.ConnectionError as e:
        print(f"Failed to connect to MongoDB: {e}")
        exit(1)

    # 불러올 파일 및 컬렉션 이름 설정
    collection_name = 'contents1'  # 예시로 설정한 컬렉션 이름
    filename = 'frame_0001.png'  # 불러올 파일 이름

    # GridFS에서 데이터 불러오기
    saliency_map_binary, img_size = load_from_gridfs(db, collection_name, filename)

    if saliency_map_binary and img_size:
        img_height, img_width = img_size
        saliency_map = np.frombuffer(saliency_map_binary, dtype=np.float32).reshape((img_height, img_width))