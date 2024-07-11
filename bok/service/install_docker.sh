#!/bin/bash

### docker 최신버전 다운로드 스크립트입니다. ###
# 필수 패키지 설치
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
# Docker GPG 키 추가
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
# Docker 저장소 추가
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
# Docker 설치
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
# Docker 설치 확인
sudo docker run hello-world
# 현재 사용자에게 Docker 권한 추가
sudo usermod -aG docker $USER && newgrp docker
# Docker 서비스 시작 및 활성화
sudo systemctl start docker
sudo systemctl enable docker
sudo service docker restart
