from flask import Flask, request, jsonify
from analysis_server import analysis
import cv2, requests, base64, logging, time
import numpy as np

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

app = Flask(__name__)
ana = analysis()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### api 통신 재시도 ###
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT"],
    backoff_factor=1
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

AGGREGATOR_URL = "http://result-aggregator-service/aggregate"
SESSION_SERVICE_URL = "http://session-service"
#################################### Session ##################################
def get_session(session_id):
    try:    
        response = http.get(f"{SESSION_SERVICE_URL}/get_session/{session_id}", timeout=5)
        response.raise_for_status()
        session_data = response.json()
        return session_data['components'].get('emotion_analysis', {})
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get session: {e}")
        return {}

def update_session(session_id, frame_num, result):
    update_data = {
        "component": "emotion_analysis",
        "data": {str(frame_num): result}
    }
    try:
        response = http.put(f"{SESSION_SERVICE_URL}/update_session/{session_id}", json=update_data)
        response.raise_for_status()
        logger.info(f"Successfully updated session for {session_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to update session: {e}")
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

    else:
        start_time = time.time()
        timeout = 5        
        try:
            while time.time() - start_time <= timeout:
                session_data = get_session(session_id)
                if len(session_data) == frame_num:
                    final_res = calc(session_data)
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("Not all frames processed")

        except TimeoutError as e:
            logger.warning(f"Timeout occurred: {e}. Proceeding with available data.")
            session_data = get_session(session_id)
            final_res = calc(session_data)

        logger.info(f"Final result: {final_res}")
        logger.info("Sending emotion data to aggregator")

        send_result(final_res, video_id, ip_address)

        return jsonify({"status": "success", "message": "Video processing completed", "final_result": final_res}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
