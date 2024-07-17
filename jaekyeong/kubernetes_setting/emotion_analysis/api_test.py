import base64, cv2, json, requests

frame = cv2.imread('a.png')
_, buffer = cv2.imencode('.jpg', frame)
frame_base64 = base64.b64encode(buffer).decode('utf-8')

# JSON 데이터 생성
data = {
    "video_id": "test_video",
    "ip_address": "127.0.0.1",
    "frame_number": 1,
    "last_frame": False,
    "frame": frame_base64
}

# JSON 데이터를 서버로 전송
response = requests.post(
    'http://192.168.58.2:30604/emotion',
    headers={'Content-Type': 'application/json'},
    data=json.dumps(data)
)
