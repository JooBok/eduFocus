### gaze_model/
* 시선추적모델 훈련 및 테스트 디렉토리입니다.

> GBR_reg_calibrate.py -> 시선추적모델 훈련 + 간단한 모델 테스트 파일    
> prediction.py -> 시선추적 서비스 테스트 파일    
> gaze_data.json -> 훈련 데이터 dump (greed serch cv 사용하여 모델 버전 관리 학습용도)

### kubernetes/
* 쿠버네티스에 사용할 이미지 빌드용 디렉토리입니다.

> session\ -> central-session 이미지 빌드 디렉토리    
> api_gateway\ -> api-gw 이미지 빌드 디렉토리    
> gaze_tracking\ -> gaze-tracking 이미지 빌드 디렉토리    
> emotion_analysis\ -> emotion-analysis 이미지 빌드 디렉토리    
> aggregator\ -> result-aggregator 이미지 빌드 디렉토리
