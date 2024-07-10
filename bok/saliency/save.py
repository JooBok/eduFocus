import os
import cv2
import json
import gridfs
import pySaliencyMap
from dotenv import load_dotenv
from pymongo import MongoClient

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
db = client['saliency_db']
fs = gridfs.GridFS(db)

input_dir = 'contents'  # 컨텐츠 이미지 파일들이 있는 디렉터리

if __name__ == '__main__':
    frame_number = 1  # 키로 사용할 초기 값

    # input_dir의 모든 파일에 대해 반복
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 이미지 파일만 처리
            input_img_path = os.path.join(input_dir, filename)
            
            # 이미지 읽기
            img = cv2.imread(input_img_path)
            if img is None:
                print(f"Failed to read {input_img_path}")
                continue  # 이미지 읽기에 실패하면 건너뛰기
            
            # 초기화
            imgsize = img.shape
            img_width = imgsize[1]
            img_height = imgsize[0]
            sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
            
            # Saliency map 만들기
            saliency_map = sm.SMGetSM(img)
            
            # saliency map이 올바르게 계산되었는지 확인
            if saliency_map is None:
                print(f"Saliency map computation failed for {filename}")
                continue
            
            # saliency map을 JSON 문자열로 변환
            saliency_map_json = json.dumps(saliency_map.tolist())
            
            # GridFS에 saliency map 저장
            file_id = fs.put(saliency_map_json.encode('utf-8'), filename=filename, frame_number=frame_number)
            
            print(f"Processed and saved saliency map for frame {frame_number} from {filename}")
            
            # 프레임 번호 증가
            frame_number += 1

    print("All saliency maps have been saved to MongoDB using GridFS")