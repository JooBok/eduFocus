# Minikube IP주소 변수 저장
MINIKUBE_IP=$(minikube ip)

# NFS 서버에 저장할 디렉터리 경로 설정
cd ~/eduFocus/bok/kubernetes/save_saliency/saliency_contents
NFS_LOCAL_PATH=$(pwd)
cd ~

# NFS 서버 설치
sudo apt install nfs-kernel-server

# NFS 서버 디렉터리 경로와 연결
sudo bash -c "echo '$NFS_LOCAL_PATH *(rw,sync,no_subtree_check,no_root_squash)' >> /etc/exports"

# 설정 저장 및 재시작
sudo systemctl start nfs-kernel-server
sudo systemctl enable nfs-kernel-server
sudo systemctl restart nfs-server
sudo systemctl status nfs-kernel-server

# NFS 트래픽을 허용하도록 방화벽 설정
sudo ufw allow frm $MINIKUBE_IP to any port nfs
sudo ufw allow frm $MINIKUBE_IP to any port 2049
sudo ufw allow frm $MINIKUBE_IP to any port 111
sudo ufw allow frm $MINIKUBE_IP to any port 20048
sudo ufw allow frm $MINIKUBE_IP to any port 875