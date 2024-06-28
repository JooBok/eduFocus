#! /bin/bash

minikube start --gpus nvidia --memory=3072 --cpus=2

minikube kubectl -- get pods -A

kubectl apply -f ./storageClass.yaml

kubectl get storageclass
