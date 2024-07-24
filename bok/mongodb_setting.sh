#!/bin/bash

# 컨텐츠 이름을 받을 인자 설정
CONTENTS_NAME="contents1"

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