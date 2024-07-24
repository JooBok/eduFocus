FROM ubuntu:22.04

# APT 패키지 리스트 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y gnupg wget

# MongoDB GPG 키 추가
RUN wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | apt-key add -

# MongoDB 패키지 저장소 추가
RUN echo "deb [ arch=amd64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/4.4 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.4.list

# APT 패키지 리스트 업데이트 및 MongoDB 서버 설치
RUN apt-get update && apt-get install -y mongodb-org

# 데이터 디렉토리 생성
RUN mkdir -p /data/db

# 기본 동작 설정
CMD ["mongod", "--bind_ip_all"]