# !bin/bash

kubectl apply -f mongodb.yaml
kubectl apply -f mongodb-express.yaml
kubectl apply -f configmap.yaml
kubectl apply -f pv.yaml
kubectl apply -f pvc.yaml
kubectl apply -f secret.yaml
