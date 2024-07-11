# Ubuntu 22.04 이미지 기반
FROM ubuntu:22.04

# 필요한 패키지 업데이트 및 설치
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && apt-get clean

# MongoDB 클라이언트 도구 설치
RUN curl -fsSL https://www.mongodb.org/static/pgp/server-4.4.asc | apt-key add - \
    && echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.4.list \
    && apt-get update \
    && apt-get install -y mongodb-org-tools \
    && apt-get clean

# 작업 디렉토리 설정
WORKDIR /app

# 실행 스크립트 복사 및 실행 권한 추가
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# 애플리케이션 실행 명령
CMD ["/app/run.sh"]