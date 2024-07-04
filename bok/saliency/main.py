# import cv2
# import matplotlib.pyplot as plt
# import pySaliencyMap

# image_name = 'milkt'
# file_name = '.jpg'

# def main():
#     # 이미지 경로
#     image = cv2.imread(image_name + file_name)
    
#     if image is None:
#         print("이미지 없음")
#         return

#     # 이미지 크기
#     image_h, image_w = image.shape[:2]
    
#     # pySaliencyMap 초기화
#     saliency = pySaliencyMap.pySaliencyMap(image_w, image_h)
    
#     # 주목성 맵 계산
#     saliency_map = saliency.SMGetSM(image)
#     binarized_map = saliency.SMGetBinarizedSM(image)
#     salient_region = saliency.SMGetSalientRegion(image)
    
#     # 주목성 맵 표시
#     plt.figure(figsize=(10, 8))
#     plt.subplot(2, 2, 1)
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title('Input Image')
#     plt.axis('off')

#     plt.subplot(2, 2, 2)
#     plt.imshow(saliency_map, cmap='gray')
#     plt.title('Saliency Map')
#     plt.axis('off')

#     plt.subplot(2, 2, 3)
#     plt.imshow(binarized_map, cmap='gray')
#     plt.title('Binarized Saliency Map')
#     plt.axis('off')

#     plt.subplot(2, 2, 4)
#     plt.imshow(cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB))
#     plt.title('Salient Region')
#     plt.axis('off')

#     plt.tight_layout()
#     plt.savefig(f'{image_name}_saliency.png')
#     print(f"이미지 저장 완료: {image_name}_saliency.png")

# if __name__ == "__main__":
#     main()

import os
import cv2
import matplotlib.pyplot as plt
import pySaliencyMap

# 이미지가 저장된 폴더와 결과를 저장할 폴더 경로
input_folder = 'test_data/contents'
output_folder = 'output'

def process_image(image_path, output_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return

    # 이미지 크기
    image_h, image_w = image.shape[:2]
    
    # pySaliencyMap 초기화
    saliency = pySaliencyMap.pySaliencyMap(image_w, image_h)
    
    # 주목성 맵 계산
    saliency_map = saliency.SMGetSM(image)
    binarized_map = saliency.SMGetBinarizedSM(image)
    salient_region = saliency.SMGetSalientRegion(image)
    
    # 주목성 맵 표시 및 저장
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(saliency_map, cmap='gray')
    plt.title('Saliency Map')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(binarized_map, cmap='gray')
    plt.title('Binarized Saliency Map')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB))
    plt.title('Salient Region')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"이미지 저장 완료: {output_path}")

def main():
    # content 폴더 내의 모든 파일 처리
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_saliency.png")
            process_image(image_path, output_path)

if __name__ == "__main__":
    main()
