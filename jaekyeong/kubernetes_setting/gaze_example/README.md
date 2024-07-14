### **작동원리**

* 인풋 = mediapipe를 통한 랜드마크 좌표
* preprocessing = 랜드마크를 통해 distance->화면과 사용자의 거리, iris_vector(x,y,z)->홍채 움직임 거리와 방향, head_rotation(x, y, z)->머리 회전을 구함
* model(Random Forest regressor) input = {iris_x, iris_y, iris_z, head_rotation_x, head_rotation_y, head_rotation_z, distance}
* model output = {gaze_x, gaze_y}

### **주요 기능들 정리**

* 모델 로딩 최적화:
  * 모델은 전역 변수로 한 번만 로드, 스레드 안전을 위해 Lock을 사용
  * 필요할 때만 모델을 로드하는 지연 로딩 방식을 사용

* session 관리
  * Redis를 사용하여 사용자 세션 정보를 저장하고 관리

* GazeBuffer
  * 최근 시선 위치를 버퍼링하고 평균을 계산

* HPA
  * cpu 사용량이 70% 넘어갈 시 새로운 pod을 생성

* LoadBalancer