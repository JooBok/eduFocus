from flask import Flask, request, jsonify
from analysis_server import analysis
import cv2, requests, base64, logging
import numpy as np
import redis, pickle

app = Flask(__name__)
ana = analysis()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGGREGATOR_URL = "http://result-aggregator-service/aggregate"
SESSION_SERVICE_URL = "http://session-service"
#################################### Session ##################################
def get_session(session_id):
    response = requests.get(f"{SESSION_SERVICE_URL}/get_session/{session_id}")
    if response.status_code == 200:
        session_data = response.json()
        return session_data['components'].get('emotion_analysis', {})
    else:
        logger.error(f"Failed to get session: {response.text}")
        return {}

def update_session(session_id, frame_num, result):
    update_data = {
        "component": "emotion_analysis",
        "data": {str(frame_num): result}
    }
    response = requests.put(f"{SESSION_SERVICE_URL}/update_session/{session_id}", json=update_data)
    if response.status_code != 200:
        logger.error(f"Failed to update session: {response.text}")

##############################################################################
def calc(final_result):
    total_frames = len(final_result)
    count = sum(1 for result in final_result.values() if result == 1)
    logging.info(f"total_frames: {total_frames}")
    logging.info(f"1's Count: {count}")
    res = count / total_frames if total_frames > 0 else 0
    res = round(res, 4) * 100
    return res

def send_result(final_result, video_id, ip_address):
    data = {
        "video_id": video_id,
        "final_score": final_result,
        "ip_address": ip_address,
        "model_type": "emotion"
    }
    response = requests.post(AGGREGATOR_URL, json=data)

    if response.status_code != 200:
        print(f"Failed to send result: {response.text}")

@app.route('/emotion', methods=['POST'])
def analyze_frame():
    data = request.json

    session_id = data['session_id']
    video_id = data['video_id']
    frame_num = data['frame_number']
    last_frame = data['last_frame']
    ip_address = data['ip_address']

    if not last_frame:
        frame_file = data['frame']
        frame_file = base64.b64decode(frame_file)
        nparr = np.frombuffer(frame_file, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result = ana.detect_face(frame)
        logging.info(f"\n=========================\nframe: {frame_num} -> {result} \n=========================")
        if result is not None:
            update_session(session_id, frame_num, result)
        logging.info(f"{ip_address} {video_id} frame {frame_num} processed")
        return jsonify({"status": "success", "message": "Frame processed"}), 200
    else:
        session_data = get_session(session_id)
        final_res = calc(session_data)
        logging.info(f"\n++++++++++++++++++++++++\nFinal Result: {final_res}\n++++++++++++++++++++++++")

        send_result(final_res, video_id, ip_address)
        logging.info(f"\n=========================\nsend emotion data to aggregator\n=========================")
        
        return jsonify({"status": "success", "message": "Video processing completed", "final_result": final_res}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)