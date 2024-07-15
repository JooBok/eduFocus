# ! bin/bash

### secret key 생성 ###
# kubectl create secret generic minio-secret \
#   --from-literal=access-key=minio-access-key \
#   --from-literal=secret-key=minio-secret-key

docker build -t pepi10/result-aggregator:latest -f Dockerfile.aggregator
docker push pepi10/result-aggregator:latest

### aggregator 배포 ###
kubectl apply -f aggregation.yaml
kubectl apply -f agg_hpa.yaml

### minio 배포 ###
kubectl apply -f minio.yaml
kubectl apply -f minio_pv.yaml
