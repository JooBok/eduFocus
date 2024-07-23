from flask import Flask, request, jsonify
from analysis_server import analysis
import cv2, requests, base64, logging, time
import numpy as np
from threading import Lock

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGGREGATOR_URL = "http://result-aggregator-service/aggregate"
SESSION_SERVICE_URL = "http://session-service"

model_lock = Lock()
ana = None
def load_model():
    global ana
    with model_lock:
        if ana is None:
            ana = analysis()
load_model()
#################################### Session ##################################
def get_session(session_id):
    try:
        response = requests.get(f"{SESSION_SERVICE_URL}/get_session/{session_id}")
        response.raise_for_status()
        session_data = response.json()
        return session_data['components'].get('emotion_analysis', {})
    except requests.RequestException as e:
        logger.error(f"Failed to get session: {str(e)}")
        return {}

def update_session(session_id, frame_num, result):
    update_data = {
        "component": "emotion_analysis",
        "data": {
            "components": {
                "emotion_analysis": {
                    str(frame_num): result
                }
            }
        }
    }
    try:
        response = requests.put(f"{SESSION_SERVICE_URL}/update_session/{session_id}", json=update_data)
        response.raise_for_status()
        logger.info(f"Successfully updated session for {session_id}")
    except requests.RequestException as e:
        if e.response is not None and e.response.status_code == 404:
            logger.warning(f"Session not found for {session_id}. Attempting to create a new session.")
            create_session(session_id)
            update_session(session_id, frame_num, result)
        else:
            logger.error(f"Failed to update session: {str(e)}")

def create_session(session_id: str) -> None:
    create_data = {
        "session_id": session_id,
        "components": {
            "emotion_analysis": {}
        }
    }
    try:
        response = requests.post(f"{SESSION_SERVICE_URL}/mk-session", json=create_data)
        response.raise_for_status()
        logger.info(f"Successfully created new session for {session_id}")
    except requests.RequestException as e:
        logger.error(f"Failed to create session: {str(e)}")
        raise
##############################################################################
def calc(final_result):
    total_frames = len(final_result)
    count = sum(1 for result in final_result.values() if result == 1)
    logger.info(f"total_frames: {total_frames}")
    logger.info(f"1's Count: {count}")
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
    try:
        response = requests.post(AGGREGATOR_URL, json=data)
        response.raise_for_status()
        logger.info("Successfully sent result to aggregator")
    except requests.RequestException as e:
        logger.error(f"Failed to send result: {str(e)}")

@app.route('/emotion', methods=['POST'])
def analyze_frame():
    data = request.json
    session_id = data['session_id']
    video_id = data['video_id']
    frame_num = data['frame_number']
    last_frame = data['last_frame']
    ip_address = data['ip_address']

    try:
        if not last_frame:
            frame_file = data['frame']
            frame_file = base64.b64decode(frame_file)
            nparr = np.frombuffer(frame_file, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            with model_lock:
                result = ana.detect_face(frame)
            
            logger.info(f"\n=========================\nframe: {frame_num} -> {result} \n=========================")
            if result is not None:
                update_session(session_id, frame_num, result)
            logger.info(f"{ip_address} {video_id} frame {frame_num} processed")
            return jsonify({"status": "success", "message": "Frame processed"}), 200
        else:
            session_data = get_session(session_id)
            final_res = calc(session_data)

            logger.info(f"Final result: {final_res}")
            logger.info("Sending emotion data to aggregator")
            send_result(final_res, video_id, ip_address)

            return jsonify({"status": "success", "message": "Video processing completed", "final_result": final_res}), 200
    except Exception as e:
        logger.error(f"Error in analyze_frame: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
