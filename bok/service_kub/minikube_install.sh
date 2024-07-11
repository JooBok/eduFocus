# 설치 스크립트 다운로드 및 실행 (리눅스)
curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
chmod +x minikube
sudo mv minikube /usr/local/bin/

# Minikube 클러스터 시작
minikube start
