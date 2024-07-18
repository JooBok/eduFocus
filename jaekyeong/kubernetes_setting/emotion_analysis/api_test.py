import base64, cv2, json, requests


for _ in range(0, 10):
    print(f'./face_frames/frame_000{_}.png')
    frame = cv2.imread(f'./face_frames/frame_000{_}.png')
    _, buffer = cv2.imencode('.png', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    f = open("base64.txt", 'a')
    f.write(frame_base64)
    f.close()

    data = {
        "video_id": "test_video",
        "ip_address": "127.0.0.1",
        "frame_number": 1,
        "last_frame": False,
        "frame": frame_base64
    }

    response = requests.post(
        'http://192.168.58.2:30604/emotion',
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )
