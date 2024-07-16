#! /bin/bash

### minikube 최신버전 다운로드 스크립트 입니다. (공식문서 참조) ### 
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64

echo 'alias kubectl="minikube kubectl --"' >> ~/.bashrc

source ~/.bashrc
