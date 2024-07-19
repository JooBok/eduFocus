#!/bin/bash

CONTENTS_NAME="contents1"

# MongoDB Pod 이름 설정
MONGODB_POD=$(kubectl get pods -l app-=mongodb -o jsonpath='{.item[0].metadata.name}')

# MongoDB에 데이터베이스 및 컬렉션 생성
kubectl exec -it $MONGODB_POD -- mongo <<EOF
use saliency_db
db.createCollection("$CONTENTS_NAME")
exit
EOF

# 컬렉션에 데이터 넣기
kubectl exec -it $MONGODB_POD -- bash -c "for FILE in /data/db/$CONTENTS_NAME/frame_*.bson; do mongorestore --db saliency_db --collection $CONTENTS_NAME ${FILE}; done"