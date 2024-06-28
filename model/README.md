# 대략적인 구조
pipeline.py파일 실행시 ADHD 분류와 집중도 측정이 동시에 이루어지도록 하자
```
/model
    |ㅡ pipeline.py
    |
    |ㅡ /input_data
    |       |
    |       |ㅡ/user_face
    |       |ㅡ/contents
    |
    |ㅡ /output_data
    |       |
    |       |ㅡ/gaze_data
    |       |ㅡ/gaze_face_data
    |
    |ㅡ /deepgaze
    |
    |
    |ㅡ /mediapipe
```

## 1. ADHD 분류
`/input_data/user_face, contents >> /deepgaze >> /output_data/gaze_data`
- pipeline.py파일 실행시 deepgaze디렉터리의 main.py가 실행되어 결과가 output_data 디렉터리의 gaze_data디렉터리에 저장됨

## 2. 집중도 측정
`/input_data/user_face >> /mediapipe >> /output_data/gaze_face_data`
- pipeline.py파일 실행시 mediapipe디렉터리의 main.py가 실행되어 결과가 output_data 디렉터리의 gaze_face_data디렉터리에 저장됨

--------
구현할 순서(mediapipe/deepgaze 모델)

1. 웹 캠으로 1분 57초짜리 얼굴 영상 mp4파일 만들기 >> ㅇ
2. 1에서 만들어진 mp4파일 1FPS단위로 이미지 만들어서 /input_data/user_id1/user_face에 넣기
3. /input_data/user_id1/contents에 있는 mp4파일 1FPS단위로 이미지 만들기
- 여기까지 하면 모델에 들어갈 input data 완성!!
4. mediapipe모델에 필요한 파이썬 파일 만들기
5. input data로 mediapipe모델 돌려서 결과 확인하기
- 여기까지 하면 mediapipe 모델 검증 완료
6. deepgaze모델에 필요한 파이썬 파일 만들기
7. input data로 deepgaze모델 돌려서 결과 확인하기
- 여기까지 하면 deepgaze 모델 검증 완료

--------
궁금한 점
- deepgaze 모델의 input과 output을 임의로 조정하여 들어갈 수 있는가??
- 

--------

<img width="743" alt="loading..." src="https://github.com/JooBok/eduFocus/model/architecture/first.png">