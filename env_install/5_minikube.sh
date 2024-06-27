#! /bin/bash

curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64

minikube start --gpus nvidia --memory=8192 --cpus=16

minikube kubectl -- get pods -A

echo 'alias kubectl="minikube kubectl --"' >> ~/.bashrc
alias kubectl="minikube kubectl --"

source ~/.bashrc

