#!/bin/bash

### Minikube 최신버전 다운로드 및 설치 스크립트 ### 

# Minikube 바이너리 다운로드
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64

# Minikube 바이너리 설치
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64

# 설치 확인
if minikube version; then
  echo "Minikube 설치가 성공적으로 완료되었습니다."
else
  echo "Minikube 설치에 실패했습니다."
  exit 1
fi

# kubectl alias 추가
echo 'alias kubectl="minikube kubectl --"' >> ~/.bashrc

# 새로운 alias를 적용하기 위해 bashrc 재로드
source ~/.bashrc

echo "Minikube 설치 및 설정이 완료되었습니다."
