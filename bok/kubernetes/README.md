## 데이터 준비
1. 현재 디렉터리에 contents 디렉터리와 saliency_contents 디렉터리를 생성한다.
2. 각 디렉터리에 contents1디렉터리를 생성한다
3. mp4파일을 contensts1 디렉터리에 넣는다
4. mp4파일을 20FPS의 이미지로 변환
    ```
    bash video_to_image.sh
    ```
## NFS 서버 설정
```
source nfs.sh
```
- 2049: NFS 서버의 기본 포트(클라이언트와 서버간의 데이터 전송을 위해 사용)
- 111: RPC(Remote Procedure Call)의 포트맵핑 서비스를 위한 포트(클라이언트가 NFS서버의 다른 RPC프로그램을 찾기 위해 사용)
- 20048: NFS 서버의 마운트 프로토콜을 위한 포트(클라이언트가 NFS서버의 파일시스템을 마운트할 때 사용)
- 875: RPC의 시간/일자 서비스를 위한 포트(NFS서버와 클라이언트 간의 시간 동기화를 위해 사용)


## MongoDB에 데이터 저장
```
source mongo.sh
```
- 셸파일에서 CONTENTS_NAME 변수값에 collection name 지정