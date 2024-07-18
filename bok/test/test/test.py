from pymongo import MongoClient
import time, json, bson
import numpy as np

MONGO_URI = 'mongodb://root:root@mongodb:27017/saliency_db?authSource=admin'
MONGO_DB = 'saliency_db'

def mongodb_client():
    return MongoClient(MONGO_URI)

def extract_saliencyMap(video_id, frame_num):
    client = mongodb_client()
    db = client[MONGO_DB]
    # collection = db[MONGO_COLLECTION]
    collection = db[video_id]
    
    # frame_num에 해당하는 첫 번째 문서를 찾음 (청크 또는 단일 문서)
    first_doc = collection.find_one({'frame_num': frame_num})
    
    if first_doc:
        if 'data' in first_doc:
            # 청크로 나누어진 데이터일 경우
            saliency_map_doc = load_bson_chunks_from_db(collection, frame_num)
        else:
            # 단일 BSON 데이터일 경우
            saliency_map_doc = first_doc

        if 'saliency_map' in saliency_map_doc:
            return np.array(saliency_map_doc['saliency_map'])
        else:
            raise ValueError(f"Error occurred: 'saliency_map' not found in the document for frame_num {frame_num}")
    else:
        raise ValueError(f"Error occurred: frame_num {frame_num} not found")

def load_bson_chunks_from_db(collection, frame_num):
    """청크로 나누어진 BSON 데이터를 MongoDB에서 불러와서 결합"""
    chunks = collection.find({'frame_num': frame_num}).sort('chunk_index')
    data_encoded = b""
    for chunk in chunks:
        data_encoded += chunk['data']
    return bson.BSON(data_encoded).decode()

print(extract_saliencyMap("contents2", 13))
