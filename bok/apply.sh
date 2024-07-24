# !bin/bash

kubectl apply -f secret.yaml
kubectl apply -f configmap.yaml
kubectl apply -f mongodb_pv.yaml
kubectl apply -f mongodb.yaml
kubectl apply -f mongodb-express.yaml