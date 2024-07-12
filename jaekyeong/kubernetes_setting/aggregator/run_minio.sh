# ! bin/bash

### secret key 생성 ###
kubectl create secret generic minio-secret \
  --from-literal=access-key=minio-access-key \
  --from-literal=secret-key=minio-secret-key

### minio 배포 ###
kubectl apply -f minio_pv.yaml
kubectl apply -f minio_pvc.yaml
kubectl apply -f minio.yaml
