from flask import Flask, request, jsonify
import requests, uuid, logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SESSION_SERVICE_URL = "http://session-service"
MODEL_URLS = {
        "emotion-analysis": "http://emotion-analysis-service/emotion",
        "gaze-tracking": "http://gaze-tracking-service/gaze",
        "blink-detector": "http://blink-detection-service/blink"
        }

def generate_session(ip_address, video_id):
    temp_session = f"{ip_address}:{video_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, temp_session))

def create_session(session_id, ip_address, video_id):
    response = requests.post(
            f"{SESSION_SERVICE_URL}/mk-session", 
            json={
                "session_id": session_id, 
                "ip_address": ip_address, 
                "video_id": video_id
                }
            )
    if response.status_code not in [200, 201]:
        raise Exception("Failed to create session - api gw")

@app.route('/process', methods = ["POST"])
def process_frame():
    data = request.json
    video_id = data["video_id"]
    ip_addr = data["ip_address"]
    frame_num = data["frame_number"]
    last_frame = data["last_frame"]

    ### generate session id ###
    session_id = generate_session(ip_addr, video_id)

    ### Ensure session id exists ###
    try:
        create_session(session_id = session_id, ip_address = ip_addr, video_id = video_id)
    except Exception as e:
        logger.error(f"Failed to create session: {str(e)}")
        return jsonify({"Error": "Failed to create session"}), 500

    data["session_id"] = session_id

    ### send data to model ###
    errors = []
    for model, url in MODEL_URLS.items():
        try:
            response = requests.post(url, json=data)
            if response.status_code != 200:
                logger.error(f"Error sending data to {model} - api gw. Status code: {response.status_code}")
                errors.append(f"Error sending data to {model}")
            else:
                logger.info(f"Sent data to {model} - api gw OK")
        except requests.RequestException as e:
            logger.error(f"Error sending data to {model} - api gw: {str(e)}")
            errors.append(f"Error sending data to {model}")

    if errors:
        return jsonify({"Errors": errors}), 500

    if last_frame:
        return jsonify({"message": "Last frame - api gw"}), 200

    return jsonify({"message": "Sent data successfully to all models - api gw"}), 200

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)
