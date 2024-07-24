
## 개요
---
시선 추적(gaze tracking)을 위한 모델을 구현하고 kubernetes로 배포하는 docker image를 만드는 디렉토리입니다.    
Flask를 통해 api화 하여 사용자 얼굴 image와 메타데이터를 json으로 받고,     
MediaPipe를 사용하여 얼굴 및 눈 랜드마크를 감지하여 preprocessing 후,     
gradient boosting regressor 모델을 사용하여 시선 좌표를 예측하는 프로세스입니다.    

## 주요 구성 요소

1. [라이브러리 및 모듈](#1-라이브러리-및-모듈)
2. [Flask](#2-flask)
3. [MongoDB](#3-mongodb)
4. [MediaPipe](#4-mediapipe)
5. [모델 설정](#5-모델-설정)
6. [주요 클래스 및 함수](#6-주요-클래스-및-함수)
7. [API 엔드포인트](#7-api-엔드포인트)

## 상세 설명

### 1. 라이브러리 및 모듈
* Flask: 웹 애플리케이션 프레임워크
*	MediaPipe: 얼굴 및 눈 랜드마크 감지
*	NumPy: 수치 연산
*	OpenCV (cv2): 이미지 처리
*	joblib: 모델 로딩
*	MongoDB: 데이터 저장 및 검색

### 2. Flask
모델 api화

### 3. MongoDB
MongoDB를 사용하여 빠르게 saliency map을 extract
'saliency_db' 데이터베이스를 사용
Collection은 video id로 설정되어 vedio id에 맞는 collection의 saliency map을 가져와 계산

### 4. MediaPipe
Media pipe를 통한 랜드마크 감지

### 5. 모델 설정
joblib을 사용하여 gaze 예측 모델(model_x, model_y)을 로드

### 6. 주요 클래스 및 함수
#### gazeDetector 클래스
시선 감지를 위한 주요 로직을 포함

#### 주요 메서드:
* process_single_frame(frame):
    * 단일 프레임에서 시선 위치를 추정 주요 함수
*	extract_saliencyMap(video_id): 
    * MongoDB에서 saliency map 데이터를 추출
*	get_or_create_gaze_detector(session_id): 
    * 세션 ID에 대한 GazeDetector를 가져오고 없으면 생성
*	get_session(session_id): 
    * 중앙 세션에서 api를 통해 세션 데이터를 가져옴
*	update_session(session_id, frame_number, gaze_data): 
    * 중앙 세션 데이터를 업데이트
*	create_session(session_id): 
    * 새 세션을 생성(update_session 함수 내에서 사용, 중앙 세션에 세션이 생성되어 있지 않으면 새로운 세션 생성하는 함수)
*	calculate_distance(iris_landmarks, image_height): 
    * 홍채 랜드마크를 기반으로 거리를 추정
*	estimate_gaze(eye_center, iris_center, estimated_distance): 
    * 시선 벡터를 추정
*	estimate_head_pose(face_landmarks): 
    * 얼굴 랜드마크를 기반으로 머리회전을 추정
*	correct_gaze_vector(gaze_vector, head_rotation): 
    * 머리 회전을 고려하여 시선 벡터를 보정
*	calculate_combined_gaze(left_gaze, right_gaze, head_rotation, distance): 
    * 최종 모델 인풋 데이터 형태로 만듦.
*	calc(gaze_points, saliency_map): 
    * 시선 위치와 saliency map을 비교하여 점수를 계산(최종 output을 return하는 함수)
*	send_result(final_result, video_id, ip_address): 
    * 최종 결과를 aggregator 서비스로 전송

### 7. API 엔드포인트
•	/gaze (POST): api로 이미지데이터와 메타데이터 POST

## 처리 흐름

1.	Api gateway에서 비디오 프레임을 /gaze 엔드포인트로 전송
2.	프레임이 처리되고 GazeDetector를 사용하여 시선 좌표 추정
    * mediapipe를 통한 랜드마크 좌표 추출
    * preprocessing = 랜드마크를 통해 distance->화면과 사용자의 거리, iris_vector(x,y,z)->홍채 움직임 거리와 방향, head_rotation(x, y, z)->머리 회전을 구함
    * model(Random Forest regressor) input = {iris_x, iris_y, iris_z, head_rotation_x, head_rotation_y, head_rotation_z, distance}
    * model output = {gaze_x, gaze_y}
3.	시선 좌표를 중앙 세션에 저장
4.	마지막 프레임 처리 시, 중앙 세션에 저장된 시선 좌표를 가져온 후 saliency map과 비교하여 최종 점수가 계산
5.	최종 결과 aggregator 서비스로 전송

## kubernetes
* HPA
  * cpu 사용량이 50% 넘어갈 시 새로운 pod을 생성

* LoadBalancer
  * 요청을 분산하여 처리