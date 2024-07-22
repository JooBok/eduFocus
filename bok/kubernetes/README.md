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

## pv.yaml 파일 수정
- 하단의 명령어를 실행하여 나온 IP를 nfs의 server에 입력
    ```
    ip route | grep eth0 | grep src | sed -n 's/.*src \([0-9\.]*\).*/\1/p'
    ```
- NFS와 연결할 로컬의 디렉터리 경로를 nfs의 path에 입력

## MongoDB 및 Express 이미지 실행
```
source apply.yaml
```

## MongoDB에 데이터 저장
- mongo.sh 파일에서 CONTENTS_NAME 변수값에 컨텐츠명(컬렉션 이름) 지정 후 하단의 명령어 실행
    ```
    source mongo.sh
    ```

## 컬렉션에 데이터 넣기
- MongoDB Pod에 접속
    ```
    kubectl exec -it $MONGODB_POD -- bash
    ```
- Pod에서 MongoDB로 데이터 이동(※ 컨텐츠명 입력 필수)
    ```
    for FILE in /data/db/<컨텐츠명>/frame_*.bson; do mongorestore --db=saliency_db --collection=<컨텐츠명> ${FILE}; done
    ```