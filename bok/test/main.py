import os
import cv2
import matplotlib.pyplot as plt
import pySaliencyMap

# 이미지가 저장된 폴더와 결과를 저장할 폴더 경로
input_folder = 'contents/1'
output_folder = 'saliency_contents/1'

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
    
    # 원본 데이터
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')
    # saliency map
    plt.subplot(2, 2, 2)
    plt.imshow(saliency_map, cmap='gray')
    plt.title('Saliency Map')
    plt.axis('off')
    # 이진화된 saliency map
    plt.subplot(2, 2, 3)
    plt.imshow(binarized_map, cmap='gray')
    plt.title('Binarized Saliency Map')
    plt.axis('off')
    # saliency region
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB))
    plt.title('Salient Region')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # content 폴더 내의 모든 파일 처리
    filenames = sorted(os.listdir(input_folder))
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_saliency.png")
            process_image(image_path, output_path)
    print(f"이미지 저장 완료: {output_path}")
    
if __name__ == "__main__":
    main()
