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
# Minikube IP 주소 변수 저장
MINIKUBE_IP=$(minikube ip)

# NFS 서버 설치
sudo apt update
sudo apt install nfs-kernel-server

# NFS 서버 설정 확인
cat /etc/exports

# NFS 서버 설정
sudo nano /etc/exports

# 하단의 내용 추가
<NFS와 연결할 로컬 디렉터리 경로> *(rw,sync,no_subtree_check,no_root_squash)

# 설정 저장 및 재시작
sudo systemctl start nfs-kernel-server
sudo systemctl enable nfs-kernel-server
sudo systemctl status nfs-kernel-server
sudo systemctl restart nfs-server

# NFS 트래픽을 허용하도록 방화벽 설정
sudo ufw allow from $MINIKUBE_IP to any port nfs
sudo ufw allow from $MINIKUBE_IP to any port 2049
sudo ufw allow from $MINIKUBE_IP to any port 111
sudo ufw allow from $MINIKUBE_IP to any port 20048
sudo ufw allow from $MINIKUBE_IP to any port 875

// 2049: NFS 서버의 기본 포트(클라이언트와 서버간의 데이터 전송을 위해 사용)
// 111: RPC(Remote Procedure Call)의 포트맵핑 서비스를 위한 포트(클라이언트가 NFS서버의 다른 RPC프로그램을 찾기 위해 사용)
// 20048: NFS 서버의 마운트 프로토콜을 위한 포트(클라이언트가 NFS서버의 파일시스템을 마운트할 때 사용)
// 875: RPC의 시간/일자 서비스를 위한 포트(NFS서버와 클라이언트 간의 시간 동기화를 위해 사용)

# 방화벽 상태 확인
sudo ufw status
```

## MongoDB 사용자 생성 및 데이터 저장
```
# MongoDB 사용자 생성
kubectl exec -it <mongo-pod-name> -- bash

mongo

use admin
db.createUser({
  user: "root",
  pwd: "root",
  roles: [ { role: "userAdminAnyDatabase", db: "admin" }, { role: "readWrite", db: "saliency_db" } ]
})

# pod 재시작
kubectl delete pod <mongodb-pod-name>

# mongodb 셸 접속
kubectl exec -it <mongodb-pod-name> -- mongo -u root -p root --authenticationDatabase admin

# mongodb 컨테이너에 있는 데이터를 mongodb에 저장
mongorestore --db <데이터베이스_이름> /data/db/<덤프_파일_또는_디렉토리>
```