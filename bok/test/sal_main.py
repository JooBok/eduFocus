# import cv2
# import json
# import pySaliencyMap

# input_img = 'milkt/frame_0001.jpg'  # 원본 이미지 파일
# saliency_json = 'saliency_map.json'  # saliency map을 저장할 JSON 파일

# # main
# if __name__ == '__main__':
#     # 이미지 읽기
#     img = cv2.imread(input_img)
#     # 초기화
#     imgsize = img.shape
#     img_width  = imgsize[1]
#     img_height = imgsize[0]
#     sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
#     # 연산
#     saliency_map = sm.SMGetSM(img)
   
#     # saliency map 계수 확인
#     print(saliency_map)
    
#     # saliency_map을 JSON으로 저장
#     saliency_map_list = saliency_map.tolist()  # numpy 배열을 리스트로 변환
#     with open(saliency_json, 'w') as f:
#         json.dump(saliency_map_list, f)
###########################
# import cv2
# import pySaliencyMap
# import json
# import os
# import gzip

# input_dir = 'milkt'  # 원본 이미지 파일들이 있는 디렉터리
# output_json_path = 'saliency_milkt/saliency_map.json.gz'  # 모든 saliency map이 저장될 JSON 파일

# # main
# if __name__ == '__main__':
#     saliency_maps = {}  # 모든 saliency map을 저장할 딕셔너리
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
            
#             # saliency map 그리기
#             saliency_map = sm.SMGetSM(img)
            
#             # saliency map이 올바르게 계산되었는지 확인
#             if saliency_map is None:
#                 print(f"Saliency map computation failed for {filename}")
#                 continue
            
#             # saliency map을 리스트로 변환하여 딕셔너리에 추가
#             saliency_maps[frame_number] = saliency_map.tolist()
            
#             print(f"Processed saliency map for frame {frame_number} from {filename}")
            
#             # 프레임 번호 증가
#             frame_number += 1

#     # 모든 saliency map을 gzip 압축 JSON으로 저장
#     with gzip.open(output_json_path, 'wt', encoding='utf-8') as f:
#         json.dump(saliency_maps, f)

#     print(f"All saliency maps have been saved to {output_json_path}")
####################################
import cv2
import pySaliencyMap
import json
import os

input_dir = 'milkt'  # 원본 이미지 파일들이 있는 디렉터리
output_dir = 'saliency_milkt'  # saliency map이 저장될 디렉터리

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
            output_json_path = os.path.join(output_dir, f'frame_{frame_number}.json')
            
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
            
            # saliency map을 JSON으로 저장
            with open(output_json_path, 'w') as f:
                json.dump(saliency_map.tolist(), f)
            
            print(f"Processed and saved saliency map for frame {frame_number} from {filename}")
            
            # 프레임 번호 증가
            frame_number += 1