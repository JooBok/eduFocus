<h1>Apache Kafka 로컬에서 실행</h1>
선행 조건 : 태블릿pc 또는 모바일 기기에 ipwebcam apk 다운로드

앱이 백그라운드에서 실행 가능하도록 설정.

![99999999999999999999](https://github.com/user-attachments/assets/03403095-1f93-4abc-9fc6-f5342014c6e8)

이 부분 확인해서 apk의 로컬 주소 확인

---------------------------------------------------
 리눅스 설치가 우선되어야 함.</br>
 Windows에서 cmd와 Powershell을 관리자 권한으로 실행.</br>
` wsl --install Ubuntu-22.04` 입력 (본 프로젝트에서는 22.04.4 버전을 사용)</br>


><h2>Apahce Kafka 설치</br>
https://kafka.apache.org/downloads</br>
3.7.0 Scala 2.13 버전 설치</br>
압축 해제</br>

><h2>리눅스로 Kafka 설치하는 법</h2>
설치된 wsl 실행 후</br>
원하는 버전의 다운로드 링크 주소 복사 후</br>
![0000](https://github.com/user-attachments/assets/0324ad39-7ce8-48d2-8bfb-1154f39c133c)</br>
 `wget https://downloads.apache.org/kafka/3.7.0/kafka_2.13-3.7.0.tgz` 실행해서 다운로드</br>
 `tar xzvf kafka_2.12-3.7.0.tgz` 실행해서 압축 해제</br> 

><h2>실행법</h2>
압축 해제된 폴더로 이동</br>
- `./bin/zookeeper-server-start.sh ./config/zookeeper.properties` 입력해서 zookeeper 실행</br>
- `./bin/kafka-server-start.sh ./config/server.properties` 입력해서 브로커(서버) 실행</br>
- 각각 다른 wsl로 실행. 즉, 2개의 wsl에 각각 zookeeper, 브로커 실행</br>
- 다시 wsl실행해서 `./bin/kafka-topics.sh --create --topic video_stream --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
` 실행해서 video-stream이라는 topic 생성.</br>
- Producer_ipwebcam.py 코드 실행
- Powershell 에서 api 호출시에는 ``` $headers = @{
     "Content-Type" = "application/json"
 }
 $body = @{
     video_id = "원하는 video_id 지정"
     max_frames = 원하는 최종프레임값 지정(초당 20프레임 데이터 전송 구성됨)
     ip_address = "10.41.0.154(ipwebcam애플리케이션 ip주소)" 
 } | ConvertTo-Json
 Invoke-RestMethod -Uri http://localhost:5000/start_stream -Method Post -Headers $headers -Body $body  ```</br>

 
 원하는 값을 할당 후에 실행하시고 Consumer_ipwebcam.py 실행하면 Kafka 동작함.
 
------
만약 실행이 되지 않는다면


wsl 에서 `vim ./config/server.properties` 실행해서 변수를 직접 설정해야함.

또는 직접 경로에 있는 파일을 클릭해서 수정하는 방법도 있음.

주석으로 처리되어 있는 **listeners=PLAINTEXT://** 을 listeners=PLAINTEXT://0.0.0.0:9092 로 수정

**num.partitions**를 num.partitions=1로 수정
