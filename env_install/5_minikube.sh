#! /bin/bash

curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64

echo 'alias kubectl="minikube kubectl --"' >> ~/.bashrc
alias kubectl="minikube kubectl --"

source ~/.bashrc

