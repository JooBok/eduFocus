from flask import Flask, request, jsonify
from analysis_server import analysis
import cv2, requests
import numpy as np
import redis, pickle

app = Flask(__name__)
ana = analysis()

AGGREGATOR_URL = "http://result-aggregator-service/aggregate"
#################################### Session ##################################
sessions = {}
redis_client = redis.Redis(host='redis-service', port=6379, db=1)

class Session:
    def __init__(self):
        self.final_result = {}
    def to_dict(self):
        return{
            'final_result':self.final_result
            }
    @classmethod
    def from_dict(cls, data):
        session = cls()
        session.final_result = data['final_result']
        return session

def get_session(session_key):
    session_data = redis_client.get(session_key)
    if session_data:
        return Session.from_dict(pickle.loads(session_data))
    return Session()
##############################################################################
def calc(final_result):
    if not final_result:
        return 0.0
    total_frames = len(final_result)
    c_frames = sum(1 for result in final_result.values() if result == 1)
    
    res = c_frames / total_frames
    return res

def send_result(final_result, video_id, ip_address):
    data = {
        "video_id": video_id,
        "final_score": final_result,
        "ip_address": ip_address
    }
    response = requests.post(AGGREGATOR_URL, json=data)

    if response.status_code != 200:
        print(f"Failed to send result: {response.text}")

@app.route('/emotion', methods=['POST'])
def analyze_frame():
    video_id = request.form['video_id']
    frame_num = int(request.form['frame_number'])
    last_frame = request.form['last_frame'].lower() == 'true'
    ip_address = request.form['ip_address']

    session_key = f"{ip_address}_{video_id}"

    session = get_session(session_key)

    if not last_frame:
        frame_file = request.form['frame']
        frame_data = frame_file.read()
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result = ana.detect_face(frame)

        if result:
            session.final_result[frame_num] = result
            
        return jsonify({"status": "success", "message": "Frame processed"}), 200
    else:
        final_res = calc(session.final_result)
        send_result(final_res, video_id, ip_address)
        return jsonify({"status": "success", "message": "Video processing completed"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
