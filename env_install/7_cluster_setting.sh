#! /bin/bash

minikube start --gpus nvidia --memory=8192 --cpus=16

minikube kubectl -- get pods -A

kubectl apply -f ./storageClass.yaml

kubectl get storageclass
