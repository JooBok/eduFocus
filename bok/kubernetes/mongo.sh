#!/bin/bash

CONTENTS_NAME="contents2"

# MongoDB Pod 이름 설정
MONGODB_POD=$(kubectl get pods -l app=mongodb -o=jsonpath='{.items[0].metadata.name}')

# MongoDB에 데이터베이스 및 컬렉션 생성
kubectl exec -it $MONGODB_POD -- mongo <<EOF
use admin
db.createUser({
	user:"root", 
	pwd:"root", 
	roles: [{role: "userAdminAnyDatabase", db: "admin"}, {role: "readWrite", db: "saliency_db"}]
	})
use saliency_db
db.createCollection("$CONTENTS_NAME")
exit
EOF

### 컬렉션에 데이터 넣기 (수정 필요) 
kubectl exec -it $MONGODB_POD -- bash -c "for FILE in /data/db/contents2/frame_*.bson; do mongorestore --db=saliency_db --collection=contents2 ${FILE}; done"
