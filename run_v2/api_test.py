import base64, cv2, json, requests

def encode(path):
    print(path)
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (640, 480))
    __, buffer = cv2.imencode('.png', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

### http:// minikube ip : api-gateway nodePort /process
gw_uri = 'http://192.168.49.2:30942/process' # $

for _ in range(1800):
    ### 디렉토리 이름 수정(통일) ###
    if _ <= 9:
        frame_base64 = encode(f'./face_frames/frame_000{_}.png') # $ 
    elif _ <= 99:
        frame_base64 = encode(f'./face_frames/frame_00{_}.png') # $
    elif _ <= 999:
        frame_base64 = encode(f'./face_frames/frame_0{_}.png') # $ 
    else:
        frame_base64 = encode(f'./face_frames/frame_{_}.png') # $

    ### metadata 수정 ###
    data = {
        "video_id": "contents2", # $
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

for _ in range(1800, 1801):
    frame_base64 = encode(f'./face_frames/frame_{_}.png')

    ### 위에 맞춰 metadata 수정 ###
    data = {
        "video_id": "contents2", # $
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
