# import os
# import cv2
# import json
# import gridfs
# import pySaliencyMap
# from dotenv import load_dotenv
# from pymongo import MongoClient

# # .env 파일 로드
# load_dotenv(dotenv_path='/app/.env')

# # 환경 변수에서 MongoDB 연결 정보 읽기
# username = os.getenv('MONGODB_USERNAME')
# password = os.getenv('MONGODB_PASSWORD')
# host = os.getenv('MONGODB_HOST')
# port = os.getenv('MONGODB_PORT')
# database = os.getenv('MONGODB_DB')

# # MongoDB 연결 설정
# client = MongoClient(f'mongodb://{username}:{password}@{host}:{port}/{database}?authSource=admin')
# db = client['saliency_db']
# fs = gridfs.GridFS(db)

# input_dir = '/app/contents/contents1'  # 컨텐츠 이미지 파일들이 있는 디렉터리

# if __name__ == '__main__':
#     frame_number = 1  # 키로 사용할 초기 값

#     # input_dir의 모든 파일에 대해 반복
#     for filename in sorted(os.listdir(input_dir)):
#         if filename.endswith('.jpg') or filename.endswith('.png'):  # 이미지 파일만 처리
#             input_img_path = os.path.join(input_dir, filename)
            
#             # 이미지 읽기
#             img = cv2.imread(input_img_path)
#             if img is None:
#                 print(f"Failed to read {input_img_path}")
#                 continue  # 이미지 읽기에 실패하면 건너뛰기
            
#             # 초기화
#             imgsize = img.shape
#             img_width = imgsize[1]
#             img_height = imgsize[0]
#             sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
            
#             # Saliency map 만들기
#             saliency_map = sm.SMGetSM(img)
            
#             # saliency map이 올바르게 계산되었는지 확인
#             if saliency_map is None:
#                 print(f"Saliency map computation failed for {filename}")
#                 continue
            
#             # saliency map을 JSON 문자열로 변환
#             saliency_map_json = json.dumps(saliency_map.tolist())
            
#             # GridFS에 saliency map 저장
#             file_id = fs.put(saliency_map_json.encode('utf-8'), filename=filename, frame_number=frame_number)
            
#             print(f"Processed and saved saliency map for frame {frame_number} from {filename}")
            
#             # 프레임 번호 증가
#             frame_number += 1

#     print("All saliency maps have been saved to MongoDB using GridFS")

############################

# import os
# import cv2
# import json
# import gridfs
# import pySaliencyMap
# from dotenv import load_dotenv
# from pymongo import MongoClient

# # .env 파일 로드
# load_dotenv(dotenv_path='/app/.env')

# # 환경 변수에서 MongoDB 연결 정보 읽기
# username = os.getenv('MONGODB_USERNAME')
# password = os.getenv('MONGODB_PASSWORD')
# host = os.getenv('MONGODB_HOST')
# port = os.getenv('MONGODB_PORT')
# database = os.getenv('MONGODB_DB')

# # MongoDB 연결 설정
# client = MongoClient(f'mongodb://{username}:{password}@{host}:{port}/{database}?authSource=admin')
# db = client[database]

# # DB에 저장할 파일들이 있는 디렉터리들 설정
# input_dirs = ['/app/contents/contents1', '/app/contents/contents2', '/app/contents/contents3']  # 여기에 추가적인 디렉토리 나열 가능

# # GridFS는 컬렉션을 명시적으로 사용할 수 없으므로 컬렉션을 다르게 설정하려면 다른 접근법을 사용해야 합니다
# def save_to_gridfs(collection_name, data, filename, frame_number):
#     fs = gridfs.GridFS(db, collection=collection_name)
#     file_id = fs.put(data, filename=filename, frame_number=frame_number)
#     return file_id

# for input_dir in input_dirs:
#     # 디렉토리 이름에서 컬렉션 이름을 추출
#     collection_name = os.path.basename(input_dir)

#     if __name__ == '__main__':
#         frame_number = 1  # 키로 사용할 초기 값

#         # input_dir의 모든 파일에 대해 반복
#         for filename in os.listdir(input_dir):
#             if filename.endswith('.jpg') or filename.endswith('.png'):  # 이미지 파일만 처리
#                 input_img_path = os.path.join(input_dir, filename)

#                 # 이미지 읽기
#                 img = cv2.imread(input_img_path)
#                 if img is None:
#                     print(f"Failed to read {input_img_path}")
#                     continue  # 이미지 읽기에 실패하면 건너뛰기

#                 # 초기화
#                 imgsize = img.shape
#                 img_width = imgsize[1]
#                 img_height = imgsize[0]
#                 sm = pySaliencyMap.pySaliencyMap(img_width, img_height)

#                 # Saliency map 만들기
#                 saliency_map = sm.SMGetSM(img)

#                 # saliency map이 올바르게 계산되었는지 확인
#                 if saliency_map is None:
#                     print(f"Saliency map computation failed for {filename}")
#                     continue

#                 # saliency map을 JSON 문자열로 변환
#                 saliency_map_json = json.dumps(saliency_map.tolist())

#                 # GridFS에 saliency map 저장
#                 try:
#                     file_id = save_to_gridfs(collection_name, saliency_map_json.encode('utf-8'), filename, frame_number)
#                     print(f"Processed and saved saliency map for frame {frame_number} from {filename} to collection {collection_name}")
#                 except Exception as e:
#                     print(f"Failed to save saliency map for frame {frame_number} from {filename}: {e}")

#                 # 프레임 번호 증가
#                 frame_number += 1

#         print(f"All saliency maps from {input_dir} have been saved to MongoDB in collection {collection_name}")

##################################

import os
import cv2
import json
import gridfs
import pySaliencyMap
from dotenv import load_dotenv
from pymongo import MongoClient

def save_to_gridfs(db, collection_name, data, filename, frame_number):
    fs = gridfs.GridFS(db, collection=collection_name)
    file_id = fs.put(data, filename=filename, frame_number=frame_number)
    return file_id

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
    client = MongoClient(f'mongodb://{username}:{password}@{host}:{port}/{database}?authSource=admin')
    db = client[database]

    # DB에 저장할 파일들이 있는 디렉터리들 설정
    input_dirs = ['/app/contents/contents1', '/app/contents/contents2', '/app/contents/contents3']  # 여기에 추가적인 디렉토리 나열 가능

    for input_dir in input_dirs:
        # 디렉토리 이름에서 컬렉션 이름을 추출
        collection_name = os.path.basename(input_dir)
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
                try:
                    file_id = save_to_gridfs(db, collection_name, saliency_map_json.encode('utf-8'), filename, frame_number)
                    print(f"Processed and saved saliency map for frame {frame_number} from {filename} to collection {collection_name}")
                except Exception as e:
                    print(f"Failed to save saliency map for frame {frame_number} from {filename}: {e}")

                # 프레임 번호 증가
                frame_number += 1

        print(f"All saliency maps from {input_dir} have been saved to MongoDB in collection {collection_name}")
