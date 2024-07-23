#!/bin/bash

# 결과 저장 디렉토리 생성
mkdir -p /home/fp/eduFocus/jaekyeong/kubernetes_setting/aggregator/storage

# result-aggregator 파드 이름 가져오기
POD_NAME=$(kubectl get pods -l app=result-aggregator -o jsonpath="{.items[0].metadata.name}")

# 파드에서 호스트로 데이터 복사
kubectl cp $POD_NAME:/tmp/storage/aggregated_data.csv /home/fp/eduFocus/jaekyeong/kubernetes_setting/aggregator/storage/aggregated_data.csv

# 복사 완료 메시지 출력
echo "Data copied from pod to host at $(date)"