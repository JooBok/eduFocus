import cv2
import matplotlib.pyplot as plt
import pySaliencyMap
import numpy as np

input_img = 'contents/milkt/frame_0001.jpg'  # 원본 이미지 파일
output_img = 'saliency_contents/milkt/frame_0001.png'  # saliency map 이미지가 저장될 경로 및 파일

# main
if __name__ == '__main__':
    # read
    img = cv2.imread(input_img)
    # initialize
    imgsize = img.shape
    img_width  = imgsize[1]
    img_height = imgsize[0]
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
    # computation
    saliency_map = sm.SMGetSM(img)
    binarized_map = sm.SMGetBinarizedSM(img)
    salient_region = sm.SMGetSalientRegion(img)
    
    # 시각화
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Input image')

    plt.subplot(2, 2, 2)
    plt.imshow(saliency_map, cmap='gray')
    plt.title('Saliency map')

    plt.subplot(2, 2, 3)
    plt.imshow(binarized_map)
    plt.title('Binarized saliency map')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB))
    plt.title('Salient region')
    
    # saliency map 계수 확인
    print("saliency_map", saliency_map)

    # 시각화 및 저장
    plt.figure(figsize=(8, 6))
    plt.imshow(saliency_map, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title('Saliency Map Heatmap')
    plt.savefig(output_img)
    plt.close()

    # 이진화 시각화
    plt.figure(figsize=(8, 6))
    plt.imshow(binarized_map, cmap='hot')
    plt.colorbar()
    plt.title('Binary Saliency Map Heatmap (threshold = 0.7)')
    plt.savefig('output.png')
    plt.close()

    cv2.destroyAllWindows()
