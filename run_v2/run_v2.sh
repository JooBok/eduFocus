# !bin/bash

kubectl apply -f session.yaml
kubectl apply -f api_gw.yaml
kubectl apply -f blink_detector.yaml
kubectl apply -f emotion.yaml
kubectl apply -f gaze_tracking.yaml
kubectl apply -f aggregation.yaml
kubectl apply -f hpa.yaml
kubectl apply -f minio.yaml
kubectl apply -f minio_pv.yaml
kubectl apply -f redis_deployment.yaml