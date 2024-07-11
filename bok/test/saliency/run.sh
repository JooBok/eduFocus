#!/bin/bash

# MongoDB 클라이언트 도구 설치
apt-get update
apt-get install -y mongodb-clients

# 데이터 다운로드 및 압축 해제
curl -L -o saliency_data.tar.gz https://github.com/JooBok/eduFocus/releases/download/v1.0/saliency_data.tar.gz
tar -xzvf saliency_data.tar.gz -C /app
rm saliency_data.tar.gz

# MongoDB에 데이터 삽입
for i in {1..26}; do
  mongoimport --host mongodb --port 27017 --db saliency_db --collection saliency_data --file /app/frame_$i.json --jsonArray
done

# 스크립트 완료 메시지
echo "Data import complete."