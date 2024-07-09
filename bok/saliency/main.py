import cv2
import json
import pySaliencyMap

input_img = 'milkt/frame_0001.jpg'  # 원본 이미지 파일
saliency_json = 'saliency_map.json'  # saliency map을 저장할 JSON 파일

# main
if __name__ == '__main__':
    # 이미지 읽기
    img = cv2.imread(input_img)
    # 초기화
    imgsize = img.shape
    img_width  = imgsize[1]
    img_height = imgsize[0]
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
    # 연산
    saliency_map = sm.SMGetSM(img)
   
    # saliency map 계수 확인
    print(saliency_map)
    
    # saliency_map을 JSON으로 저장
    saliency_map_list = saliency_map.tolist()  # numpy 배열을 리스트로 변환
    with open(saliency_json, 'w') as f:
        json.dump(saliency_map_list, f)
