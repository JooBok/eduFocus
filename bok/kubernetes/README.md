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
# 쿠버네티스 노드 IP 주소 확인
kubectl get nodes -o wide

# NFS 서버 설치
sudo apt update
sudo apt install nfs-kernel-server

# NFS 서버 설정 확인
cat /etc/exports

# NFS 서버 설정
sudo nano /etc/exports

# 하단의 내용 추가
/home/bok/eduFocus/bok/service_kub/application *(rw,sync,no_subtree_check,no_root_squash)
/home/bok/eduFocus/bok/service_kub/contents *(rw,sync,no_subtree_check,no_root_squash)

# 설정 저장 및 재시작
sudo systemctl start nfs-kernel-server
sudo systemctl enable nfs-kernel-server
sudo systemctl status nfs-kernel-server

# NFS 트래픽을 허용하도록 방화벽 설정
sudo ufw allow from <Kubernetes 노드 IP 주소> to any port nfs
sudo ufw allow from <Kubernetes 노드 IP 주소> to any port 2049
sudo ufw allow from <Kubernetes 노드 IP 주소> to any port 111
sudo ufw allow from <Kubernetes 노드 IP 주소> to any port 20048
sudo ufw allow from <Kubernetes 노드 IP 주소> to any port 875

# 방화벽 상태 확인
sudo ufw status
```