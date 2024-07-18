import base64, cv2, json, requests


for _ in range(0, 26):
    if _ < 10:
        print(f'./face_frames/frame_000{_}.png')
        frame = cv2.imread(f'./face_frames/frame_000{_}.png')
        frame = cv2.resize(frame, (640, 480))
        __, buffer = cv2.imencode('.png', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
    elif _ >= 10:
        frame = cv2.imread(f'./face_frames/frame_00{_}.png')
        frame = cv2.resize(frame, (640, 480))
        __, buffer = cv2.imencode('.png', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

    f = open("base64.txt", 'a')
    f.write(frame_base64)
    f.close()

    data = {
        "video_id": "contents2",
        "ip_address": "127.0.0.1",
        "frame_number": _,
        "last_frame": False,
        "frame": frame_base64
    }

    response = requests.post(
        'http://192.168.49.2:30659/gaze',
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )

print("##########################################################################")

for _ in range(26, 27):
    print(f'./face_frames/frame_00{_}.png')
    frame = cv2.imread(f'./face_frames/frame_00{_}.png')
    frame = cv2.resize(frame, (640, 480))
    _, buffer = cv2.imencode('.png', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    f = open("base64.txt", 'a')
    f.write(frame_base64)
    f.close()

    data = {
        "video_id": "contents2",
        "ip_address": "127.0.0.1",
        "frame_number": _,
        "last_frame": True,
        "frame": frame_base64
    }

    response = requests.post(
        'http://192.168.49.2:30659/gaze',
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )
