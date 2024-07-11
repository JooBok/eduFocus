from flask import Flask, request, jsonify
from util.analysis_realtime import analysis
import cv2, requests
import numpy as np

app = Flask(__name__)
ana = analysis()

AGGREGATOR_URL = "http://result-aggregator-service/aggregate"
sessions = {}

class Session:
    def __init__(self):
        self.final_result = {}

def calc(final_result):
    if not final_result:
        return 0.0
    total_frames = len(final_result)
    c_frames = sum(1 for result in final_result.values() if result == 1)
    
    res = c_frames / total_frames
    return res

def send_result(final_result, video_id):
    data = {
        "video_id": video_id,
        "final_score": final_result
    }
    response = requests.post(AGGREGATOR_URL, json=data)

    if response.status_code != 200:
        print(f"Failed to send result: {response.text}")

@app.route('/emotion', methods=['POST'])
def analyze_frame():
    video_id = request.form['video_id']
    frame_num = int(request.form['frame_number'])
    last_frame = request.form['last_frame'].lower() == 'true'

    ip_address = request.remote_addr
    session_key = f"{ip_address}_{video_id}"

    if session_key not in sessions:
        sessions[session_key] = Session()

    session = sessions[session_key]

    if not last_frame:
        frame_file = request.files['frame']
        frame_data = frame_file.read()
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result = ana.detect_face(frame)

        if result:
            session.final_result[frame_num] = result

        return jsonify({"status": "success", "message": "Frame processed"}), 200
    else:
        final_res = calc(session.final_result)
        send_result(final_res, video_id)
        del sessions[session_key]
        return jsonify({"status": "success", "message": "Video processing completed"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)