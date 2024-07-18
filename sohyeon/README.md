## 깜빡임 감지를 이용한 집중도 측정

눈 깜빡임 비율(EAR)을 사용하여 깜빡임을 감지하고 깜빡임 빈도를 기반으로 집중도 수준을 측정

├── BlinkDetector
│   ├── blink_detector.py          # 깜빡임을 감지하고 집중도를 계산하는 BlinkDetector 클래스
│   ├── Dockerfile.blinkdetector   # 애플리케이션을 컨테이너화하기 위한 Dockerfile
│   ├── blink_detector.yaml        # 여러 서비스를 실행하기 위한 Docker Compose 파일
│   └── requirements.txt           # Python 종속성 목록
├── test
│   ├── concentration_measurement_real.py  # 웹캠을 사용하여 실시간으로 집중도를 측정하는 파일
│   ├── concentration_measurement_video.py # 비디오 파일에서 집중도를 측정하는 파일
│   └── version_1.py                      # 테스트용 초기 버전 스크립트




## 집중도 계산 방법

본 프로젝트에서 집중도를 계산하기 위해 다음과 같은 공식을 사용:

### 집중도 계산 공식
\[
\text{집중도} = \frac{1}{e_c}
\]

### 집중 점수 계산 공식
\[
\text{집중 점수} = \sum_{n=1}^{m} (\text{집중도}_n \times w_i)
\]

### 집중도 e_c 계산 공식
\[
e_c = \frac{c_l e_l + c_r e_r}{c_l + c_r} \quad \text{for} \quad c_l + c_r > 0
\]

### 집중도 계산 설명

- **e_c**: 왼쪽과 오른쪽 눈의 EAR(눈 깜빡임 비율) 값을 결합한 집중도 지표.
- **c_l, c_r**: 각 눈의 감지 신뢰도 값.
- **e_l, e_r**: 각 눈의 EAR 값.
- **w_i**: 시간에 따른 가중치.

이를 통해 각 프레임에서의 집중도를 계산하고, 모든 프레임의 집중도를 가중 평균하여 최종 집중 점수를 산출한다. 
이 방법으로 일정 시간 동안의 집중 상태를 종합적으로 평가할 수 있음.

### 작동 원리

1. **BlinkDetector 클래스**:
   - MediaPipe의 FaceMesh를 사용하여 얼굴 랜드마크를 감지.
   - 눈 깜빡임 비율(EAR)을 계산하여 깜빡임 이벤트를 결정.
   - 깜빡임 빈도 및 다른 지표를 기반으로 집중도 수준을 계산.
  
## 논문 출처
본 프로젝트에서 사용된 집중도 계산 방법은 다음 논문을 참고하였음:
 - ASK 2022 학술발표대회 논문집 (29권 1호), 얼굴 인식 및 눈 깜빡임을 활용한 집중력 수치화 기법, 장환곤, 박성철, 나상우, 김민, 이영재, 김영종, 숭실대학교 소프트웨어학부
