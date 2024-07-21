from flask import Flask, request, jsonify
import requests, uuid

app = Flask(__name__)

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
    create_session(session_id, ip_addr, video_id)

    data["session_id"] = session_id

    ### send data to model ###
    for model, url in MODEL_URLS.item():
        response = requests.post(url, json = data)
        if response.status_code != 200:
            return jsonify({"Error": f"Error send data to {model} - api gw"}), 500

    if last_frame:
        return jsonify({"message": "Last frame - api gw"}), 200
    return jsonify({"message": "send data successfully - api gw"}), 200

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)
