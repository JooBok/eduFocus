import base64, cv2, json, requests, os

def encode(path):
    print(path)
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (640, 360))
    __, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

directory = './face_frames'

### http:// minikube ip : api-gateway nodePort /process
gw_uri = 'http://192.168.49.2:30942/process' # $

for _ in range(1, 300):
    ### 디렉토리 이름 수정(통일) ###
    file_path = os.path.join(directory, f'frame_{_:04d}.jpg')
    frame_base64 = encode(file_path)
    ### metadata 수정 ###
    data = {
        "video_id": "contents1", # $
        "ip_address": "127.0.0.1", # $
        "frame_number": _,
        "last_frame": False,
        "frame": frame_base64
    }

    response = requests.post(
        gw_uri,
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )
print("##########################################################################")

for _ in range(300, 301):
    file_path = os.path.join(directory, f'frame_{_:04d}.jpg')
    frame_base64 = encode(file_path)

    ### 위에 맞춰 metadata 수정 ###
    data = {
        "video_id": "contents1", # $
        "ip_address": "127.0.0.1", # $
        "frame_number": _,
        "last_frame": True,
        "frame": frame_base64
    }

    response = requests.post(
        gw_uri,
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )
