Apache Kafka 로컬에서 실행
---------------------------------------------------
> 리눅스 설치가 우선되어야 함.</br>
> Windows에서 cmd와 Powershell을 관리자 권한으로 실행.</br>
>` wsl --install Ubuntu-22.04` 입력 (본 프로젝트에서는 22.04.4 버전을 사용)</br>


><h2>Apahce Kafka 설치</h2></br>
>https://kafka.apache.org/downloads</br>
>3.7.0 Scala 2.13 버전 설치(소스였나?)</br>
>압축 해제</br>

><h2>리눅스로 Kafka 설치하는 법</h2>
>설치된 wsl 실행 또는 Powershell 실행 후</br>
>원하는 버전의 다운로드 링크 주소 복사 후</br>
![0000](https://github.com/user-attachments/assets/0324ad39-7ce8-48d2-8bfb-1154f39c133c)</br>
> `wget https://downloads.apache.org/kafka/3.7.0/kafka_2.13-3.7.0.tgz` 실행해서 압축 해제</br>

><h2>실행법</h2>
>압축 해제된 폴더로 이동</br>
`./bin/zookeeper-server-start.sh ./config/zookeeper.properties` 입력해서 zookeepr 실행
