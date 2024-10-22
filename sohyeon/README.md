## 깜빡임 감지를 이용한 집중도 측정

눈 깜빡임 비율(EAR)을 사용하여 깜빡임을 감지하고 깜빡임 빈도를 기반으로 집중도 수준을 측정

---

## 구조
├── BlinkDetector
- `blink_detector.py`: 깜빡임을 감지하고 집중도를 계산하는 `BlinkDetector` 클래스가 포함된 파일.
- `Dockerfile.blinkdetector`: 애플리케이션을 컨테이너화하기 위한 Dockerfile.
- `blink_detector.yaml`: 여러 서비스를 실행하기 위한 Docker Compose 파일.
- `requirements.txt`: Python 종속성 목록.

├── test
- `concentration_measurement_real.py`: 웹캠을 사용하여 실시간으로 집중도를 측정하는 파일.
- `concentration_measurement_video.py`: 비디오 파일에서 집중도를 측정하는 파일.
- `version_1.py`: 테스트용 초기 버전 스크립트.

---

## 집중도 계산 방법

본 프로젝트에서 집중도를 계산하기 위해 다음과 같은 공식을 사용:

### 집중도 e_c 계산 공식
<p align="center">
<img width="40%" alt="스크린샷 2024-07-03 140805" src="https://github.com/user-attachments/assets/dadc09fa-dddb-4c10-b365-db5e8fad616e"> 
</p>

### 집중도 계산 공식
<p align="center">
<img width="15%" alt="스크린샷 2024-07-05 103941" src="https://github.com/user-attachments/assets/d0cea5da-f5da-4a6d-9089-f73c29f60119"> 
</p>

### 집중 점수 계산 공식
<p align="center">
<img width="25%" alt="스크린샷 2024-07-09 131713" src="https://github.com/user-attachments/assets/f3a3c2e0-0c50-4d4e-8f73-d592d87201b6"> 
</p>

---

## 집중도 계산 과정

- **e_c**: 왼쪽과 오른쪽 눈의 EAR(눈 깜빡임 비율) 값을 결합한 집중도 지표.
- **c_l, c_r**: 각 눈의 감지 신뢰도 값.
- **e_l, e_r**: 각 눈의 EAR 값.
- **w_i**: 시간에 따른 가중치.

1. 눈 깜빡임 비율(EAR) 계산:
   - 눈 감은 값은 EAR 값으로 대체된다.
   - EAR은 눈의 특정 랜드마크 간 거리 비율로 정의되며, 눈이 깜빡일 때 EAR 값이 증가한다.
   - 이를 통해 눈 깜빡임 정도를 정량화하여 신뢰도 있는 집중도를 도출할 수 있다.


2. 눈 감은 값의 신뢰도:
   - Mediapipe는 얼굴 랜드마크를 감지할 때 각 랜드마크의 탐지 신뢰도, 존재 신뢰도, 추적 신뢰도를 제공한다.
   - Mediapipe에서 제공하는 신뢰도 값들의 평균으로 대체된다.
   
  
3. 깜빡임 감지:
   - Mediapipe를 사용하여 눈 랜드마크를 감지하여 계산되고 임계값을 설정하여 EAR 값이 이 임계값을 넘으면 눈이 감긴 것으로 간주한다.
   
4. 집중도 계산
   - 각 프레임에서 EAR 값을 사용하여 집중도를 계산
   - 집중도는 EAR 값과 감지 신뢰도(confidence) 값을 결합하여 계산
   - 집중도는 e_c 값의 역수로 정의되며, 이는 EAR값이 작을수록 (즉, 눈을 감지 않았을 때)집중도가 높아지는 특성을 가진다.

이를 통해 각 프레임에서의 집중도를 계산하고, 모든 프레임의 집중도를 가중 평균하여 최종 집중 점수를 산출한다. 
이 방법으로 일정 시간 동안의 집중 상태를 종합적으로 평가할 수 있음.

---

## 집중 점수 계산 과정

1. 프레임별 집중도 저장:
   - 각 프레임의 EAR 값을 모아서 시퀀스를 구성하고, 각 프레임에 대한 가중치는 인덱스가 증가할수록 감소하도록 설정한다.
   - 이를 통해 초반 프레임에 더 높은 가중치를 부여하여 집중도를 계산한다. 프레임 번호를 이용하여 최대 프레임 수를 계산한다

2. 시간 가중치 적용:
   - 시간 가중치는 실제 시간을 대신하여 프레임을 이용하여 계산됨
  
3. 최종 집중 점수 계산:
   - EAR 시퀀스와 시간 가중치를 사용하여 집중점수를 계산
   
  
---
     
## 작동 원리

1. **BlinkDetector 클래스**:
   - MediaPipe의 FaceMesh를 사용하여 얼굴 랜드마크를 감지.
   - 눈 깜빡임 비율(EAR)을 계산하여 깜빡임 이벤트를 결정.
   - 깜빡임 빈도 및 다른 지표를 기반으로 집중도 수준을 계산.

---
  
## 논문 출처
본 프로젝트에서 사용된 집중도 계산 방법은 다음 논문을 참고하였음:
 - ASK 2022 학술발표대회 논문집 (29권 1호), 얼굴 인식 및 눈 깜빡임을 활용한 집중력 수치화 기법, 장환곤, 박성철, 나상우, 김민, 이영재, 김영종, 숭실대학교 소프트웨어학부
