#! /bin/bash

### minikube gpu 실행과 스탠다드 스토리지 클래스를 생성하는 스크립트입니다. ###
### 컴퓨터 사양에 맞게 재구성해주세요 ###
minikube start --memory=5120 --cpus=8

minikube kubectl -- get pods -A

minikube tunnel &> /dev/null &
