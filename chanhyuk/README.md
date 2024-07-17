Apache Kafka 로컬에서 실행
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
`./bin/zookeeper-server-start.sh ./config/zookeeper.properties` 입력해서 zookeeper 실행</br>
`./bin/kafka-server-start.sh ./config/server.properties` 입력해서 브로커(서버) 실행</br>
각각 다른 wsl로 실행. 즉, 2개의 wsl에 각각 zookeeper, 브로커 실행</br>
다시 wsl실행해서 `bin/kafka-topics.sh --create --topic video_stream --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
` 실행해서 video-stream이라는 topic 생성.</br>
