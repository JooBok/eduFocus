import cv2
import pySaliencyMap
import bson
import os

input_dir = 'contents/contents1'  # 원본 이미지 파일들이 있는 디렉터리
output_dir = 'saliency_contents/contents1'  # saliency map이 저장될 디렉터리

# output_dir이 존재하지 않으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# main
if __name__ == '__main__':
    frame_number = 1  # 키로 사용할 초기 값

    # input_dir의 모든 파일에 대해 반복
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 이미지 파일만 처리
            input_img_path = os.path.join(input_dir, filename)
            output_bson_path = os.path.join(output_dir, f'frame_{frame_number}.bson')
            
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
            
            # 연산
            saliency_map = sm.SMGetSM(img)
            
            # saliency map이 올바르게 계산되었는지 확인
            if saliency_map is None:
                print(f"Saliency map computation failed for {filename}")
                continue
            
            # saliency map을 BSON으로 저장
            with open(output_bson_path, 'wb') as f:
                f.write(bson.BSON.encode({'saliency_map': saliency_map.tolist()}))
            
            print(f"Processed and saved saliency map for frame {frame_number} from {filename}")
            
            # 프레임 번호 증가
            frame_number += 1
