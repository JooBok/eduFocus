import cv2
import pySaliencyMap
import bson
import os
import math
import sys

MAX_BSON_SIZE = 16 * 1024 * 1024  # 16MB

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

def main(input_dir, output_dir):
    """"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_number = 1

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_img_path = os.path.join(input_dir, filename)
            output_bson_path = os.path.join(output_dir, f'frame_{frame_number}')
            
            img = cv2.imread(input_img_path)
            if img is None:
                print(f"Failed to read {input_img_path}")
                continue
            
            img_resized = cv2.resize(img, (640, 480))
            
            imgsize = img_resized.shape
            img_width = imgsize[1]
            img_height = imgsize[0]
            sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
            
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
            
            frame_number += 1

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <directory_name>")
        sys.exit(1)
    
    dir_name = sys.argv[1]
    input_dir = f'contents/{dir_name}'
    output_dir = f'saliency_contents/{dir_name}'
    
    main(input_dir, output_dir)