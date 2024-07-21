import base64, cv2, json, requests

def encode(path):
    print(path)
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (640, 480))
    __, buffer = cv2.imencode('.png', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

gw_uri = 'http://api-gateway-service/process'

for _ in range(20):
    if _ < 10:
        frame_base64 = encode(f'./face_frames/frame_000{_}.png')
    elif _ >= 10:
        frame_base64 = encode(f'./face_frames/frame_00{_}.png')

    data = {
        "video_id": "contents2",
        "ip_address": "127.0.0.1",
        "frame_number": _,
        "last_frame": False,
        "frame": frame_base64
    }

    response = requests.post(
        gw_url,
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )
print("##########################################################################")

for _ in range(20, 21):
    frame_base64 = encode(f'./face_frames/frame_00{_}.png')

    data = {
        "video_id": "contents2",
        "ip_address": "127.0.0.1",
        "frame_number": _,
        "last_frame": True,
        "frame": frame_base64
    }

    response = requests.post(
        gw_url,
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )
