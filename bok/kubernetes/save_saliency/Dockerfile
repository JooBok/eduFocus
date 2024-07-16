FROM Ubuntu:22.04

# 필요한 패키지 업데이트 및 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && apt-get clean

# 작업 디렉토리 설정
WORKDIR /app

# Python 패키지 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# App 소스코드 복사
COPY . .

# 애플리케이션 실행 명령
ENTRYPOINT ["python", "main.py"]