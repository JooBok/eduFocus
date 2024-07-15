# Ubuntu 22.04 이미지 기반
FROM ubuntu:22.04

# 필요한 패키지 업데이트 및 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# 작업 디렉토리 설정
WORKDIR /app

# Python 패키지 설치
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# 앱 소스 코드 복사
COPY ./application /app

# .env 파일 복사
COPY .env /app/.env

# 애플리케이션 실행 명령
CMD ["python3", "main.py"]