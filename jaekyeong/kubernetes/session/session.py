from flask import Flask, request, jsonify
import redis, json, logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

app = Flask(__name__)
redis_client = redis.Redis(host = "redis-service", port = 6379)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### 세션 파기 (대기시간 5분) ###
SESSION_EXPIRY = timedelta(minutes = 5)

@app.route('/mk-session', methods = ['POST'])
def mk_session() -> Tuple[Any, int]:
    ### api gw로 부터 받는 data ###
    data = request.json
    logger.info(f" session mk-session -> data: {data}")
    session_id = data["session_id"]
    ip_address = data["ip_address"]
    video_id = data["video_id"]

    ### session check ###
    if redis_client.exists(f"session:{session_id}"):
        return jsonify({"message": "Session Already Exist"}), 200

    ### session make ###
    session_data: Dict[str, Any] = {
            "session_id": session_id,
            "ip_address": ip_address,
            "video_id": video_id,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "components": {
                "gaze_tracking": {},
                "blink_detection": {},
                "emotion_analysis": {},
            }
        }
    ### session store ###
    redis_client.setex(
            f"session:{session_id}",
            int(SESSION_EXPIRY.total_seconds()),
            json.dumps(session_data))
    return jsonify({"session_id": session_id}), 201

@app.route('/get_session/<session_id>', methods = ['GET'])
def get_session(session_id: str) -> Tuple[Any, int]:
    """
    session 확인용 함수와 api endpoint
    """
    session_data = redis_client.get(f"session:{session_id}")
    if session_data:
        return jsonify(json.loads(session_data)), 200
    return jsonify({'error': 'Session Not Found'}), 404

@app.route('/update_session/<session_id>', methods = ['PUT'])
def update_session(session_id: str) -> Tuple[Any, int]:
    """
    session에 gaze, blink, emotion의 data 넣는 함수와 api endpoint
    """
    component = request.json.get('component')
    data = request.json.get('data', {})

    session_data = redis_client.get(f"session:{session_id}")
    if session_data:
        session = json.loads(session_data)

        if 'components' in data and component in data['components']:
            session['components'][component].update(data['components'][component])
        else:
            session['components'][component].update(data)

        session['last_accessed'] = datetime.now().isoformat()

        redis_client.setex(
            f"session:{session_id}",
            int(SESSION_EXPIRY.total_seconds()),
            json.dumps(session))

        return jsonify({'message': 'Session Updated Successfully'}), 200
    return jsonify({'error': 'Session Not Found'}), 404

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)