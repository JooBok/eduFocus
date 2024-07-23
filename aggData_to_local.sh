#!/bin/bash

### 결과 저장 디렉토리 생성 (위치 원하는 곳으로 변경)###
mkdir -p /home/fp/eduFocus/jaekyeong/storage

### result-aggregator 파드 이름 가져오기 ###
POD_NAME=$(kubectl get pods -l app=result-aggregator -o jsonpath="{.items[0].metadata.name}")

### 파드에서 호스트로 데이터 복사 ###
kubectl cp $POD_NAME:/tmp/storage/aggregated_data.csv /home/fp/eduFocus/jaekyeong/storage/aggregated_data.csv

echo "Data copied from pod to host at $(date)"