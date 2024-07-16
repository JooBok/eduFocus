## 깜빡임 감지를 이용한 집중도 측정

눈 깜빡임 비율(EAR)을 사용하여 깜빡임을 감지하고 깜빡임 빈도를 기반으로 집중도 수준을 측정

## 구조
├── BlinkDetector
- `blink_detector.py`: 깜빡임을 감지하고 집중도를 계산하는 `BlinkDetector` 클래스가 포함된 파일.
- `Dockerfile.blinkdetector`: 애플리케이션을 컨테이너화하기 위한 Dockerfile.
- `blink_detector.yaml`: 여러 서비스를 실행하기 위한 Docker Compose 파일.
- `requirements.txt`: Python 종속성 목록.

├── test
- `concentration_measurement_real.py`: 웹캠을 사용하여 실시간으로 집중도를 측정하는 파일.
- `concentration_measurement_video.py`: 비디오 파일에서 집중도를 측정하는 파일.
- 'version_1.py'

├── 

├── 








## 작동 원리
- BlinkDetector 클래스
BlinkDetector 클래스는 MediaPipe의 FaceMesh를 사용하여 얼굴 랜드마크를 감지하고, 눈 깜빡임 이벤트를 결정하기 위해 눈 깜빡임 비율(EAR)을 계산합니다. 
그런 다음 깜빡임 빈도를 기반으로 집중도 수준을 계산합니다.
- 비디오 집중도 측정
concentration_measurement_video.py의 ConcentrationMeasurement 클래스는 비디오 파일을 처리하고 프레임마다 집중도 수준을 계산하여 결과를 기록합니다.
- 실시간 집중도 측정
concentration_measurement_real.py의 ConcentrationApp 클래스는 tkinter GUI를 사용하여 웹캠을 통해 실시간으로 집중도 측정을 시작하고 종료합니다.

## 설정
- EAR 임계값 및 연속 프레임 수
BlinkDetector 클래스에서 EAR 임계값과 깜빡임을 감지할 연속 프레임 수를 조정
