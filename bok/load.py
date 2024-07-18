import bson
import os

output_dir = 'saliency_contents/contents2'  # 청크로 나누어 저장된 BSON 파일들이 있는 디렉터리

def load_bson_chunks(base_path):
    """ 청크로 나누어 저장된 BSON 파일들을 불러와 재구성 """
    chunks = []
    i = 0
    while True:
        chunk_path = f"{base_path}_chunk_{i}.bson"
        if not os.path.exists(chunk_path):
            break
        with open(chunk_path, 'rb') as f:
            chunks.append(f.read())
        i += 1
    
    # 청크들을 결합하여 원래의 BSON 데이터로 복원
    if chunks:
        return bson.BSON(b''.join(chunks)).decode()
    else:
        return None

def load_data_for_frame(frame_number):
    """ 특정 frame_number에 맞는 데이터를 불러오기 """
    base_path = os.path.join(output_dir, f'frame_{frame_number}')
    
    if os.path.exists(f"{base_path}.bson"):
        # 단일 파일로 저장된 BSON 불러오기
        with open(f"{base_path}.bson", 'rb') as f:
            data = bson.BSON(f.read()).decode()
    else:
        # 청크로 나누어 저장된 BSON 불러오기
        data = load_bson_chunks(base_path)
    
    return data

# main
if __name__ == '__main__':
    frame_number = int(input("Enter frame number to load: "))  # 사용자로부터 frame_number 입력받기

    data = load_data_for_frame(frame_number)
    
    if data is not None:
        print(f"Loaded data for frame {frame_number}:")
        print(data)
    else:
        print(f"Failed to load data for frame {frame_number}")
