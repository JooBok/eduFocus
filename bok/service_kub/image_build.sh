# Docker Hub 로그인
$ docker login
# Docker 이미지를 빌드
$ docker build -t joobok/mongo:v1.0 .
# 빌드된 이미지 테스트
$ docker run -d -p 8000:8000 joobok/mongo:v1.0
# MongoDB 이미지 태깅
$ docker tag mongo:v1.0 joobok/mongo:v1.0
# 이미지 푸시
$ docker push joobok/mongo:v1.0