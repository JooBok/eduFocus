import cv2
import pySaliencyMap
import bson
import os
import math

input_dir = 'contents/contents2'  # 원본 이미지 파일들이 있는 디렉터리
output_dir = 'saliency_contents/contents2'  # saliency map이 저장될 디렉터리
MAX_BSON_SIZE = 16 * 1024 * 1024  # 16MB

# output_dir이 존재하지 않으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_bson_chunks(data, output_path, max_chunk_size=MAX_BSON_SIZE):
    """ 데이터를 청크로 나누어 BSON 파일로 저장 """
    data_encoded = bson.BSON.encode(data)
    total_size = len(data_encoded)
    num_chunks = math.ceil(total_size / max_chunk_size)
    chunk_size = math.ceil(len(data_encoded) / num_chunks)
    
    for i in range(num_chunks):
        chunk_data = data_encoded[i*chunk_size:(i+1)*chunk_size]
        chunk_output_path = f"{output_path}_chunk_{i}.bson"
        with open(chunk_output_path, 'wb') as f:
            f.write(chunk_data)

# main
if __name__ == '__main__':
    frame_number = 1  # 키로 사용할 초기 값

    # input_dir의 모든 파일에 대해 반복
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 이미지 파일만 처리
            input_img_path = os.path.join(input_dir, filename)
            output_bson_path = os.path.join(output_dir, f'frame_{frame_number}')
            
            # 이미지 읽기
            img = cv2.imread(input_img_path)
            if img is None:
                print(f"Failed to read {input_img_path}")
                continue  # 이미지 읽기에 실패하면 건너뛰기
            
            # 이미지 크기 변경
            img_resized = cv2.resize(img, (640, 480))
            
            # 초기화
            imgsize = img_resized.shape
            img_width = imgsize[1]
            img_height = imgsize[0]
            sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
            
            # 연산
            saliency_map = sm.SMGetSM(img_resized)
            
            # saliency map이 올바르게 계산되었는지 확인
            if saliency_map is None:
                print(f"Saliency map computation failed for {filename}")
                continue
            
            # saliency map과 파일명을 BSON으로 저장
            bson_data = {
                'frame_num': frame_number,
                'saliency_map': saliency_map.tolist()
            }
            
            # 데이터가 16MB를 초과할 경우 청크로 나누어 저장
            if len(bson.BSON.encode(bson_data)) > MAX_BSON_SIZE:
                save_bson_chunks(bson_data, output_bson_path)
            else:
                with open(f"{output_bson_path}.bson", 'wb') as f:
                    f.write(bson.BSON.encode(bson_data))
            
            print(f"Processed and saved saliency map for frame {frame_number} from {filename}")
            
            # 프레임 번호 증가
            frame_number += 1