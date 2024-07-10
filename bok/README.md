# 개인 작업공간

## saliency 디렉터리
| 파일명              | 설명                                                                 |
|-------------------|--------------------------------------------------------------------|
| **docker-compose.yml**| MongoDB와 Mongo-Express 컨테이너로 구성                               |
| **1_install_docker.sh**    | Ubuntu에 최신버전의 Docker, Docker-compose를 설치하는 셸 파일                    |
| **2_install_nvidia_docker.sh** | docker nvidia 최신버전을 설치하는 셸 파일                         |
| **3_video_to_image.sh** | mp4 파일을 20FPS로 이미지로 저장하는 셸 파일                         |
| **save.py**           | contents 디렉터리의 모든 이미지(png, jpg)를 saliency map을 그려 MongoDB에 저장하는 파일 |
| **load.py**           | MongoDB에서 saliency map을 불러오는 파일                              |
| **pySaliencyMap.py**  | Saliency map을 만드는 함수들이 정의되어 있는 파일                      |
| **pySaliencyMapDef.py**| Saliency map을 그리기 위한 변수들이 정의되어 있는 파일                |
---
[**추가로 구성해야 할 내용**]
- /contents: 컨텐츠가 20FPS의 이미지로 저장된 파일들이 있는 디렉터리
- .env: MongoDB의 민감정보들

## .env 파일 설정
```
MONGODB_USERNAME=<your username>
MONGODB_PASSWORD=<your password>
MONGODB_HOST=localhost
MONGODB_PORT=<your mongodb port>
MONGODB_EXPRESS_PORT=<your mongodb-express port>
MONGODB_DB=<your db name>
ME_CONFIG_BASICAUTH_USERNAME=<your mongo-express username>
ME_CONFIG_BASICAUTH_PASSWORD=<your mongo-express password>

```

## 예시 데이터 사용 방법
```
curl -L -o saliency_data.tar.gz https://github.com/JooBok/eduFocus/releases/download/v1.0/saliency_data.tar.gz
```

## 실제 데이터 사용하는 경우
1. Saliency map을 만들 컨텐츠 영상 mp4파일을 contents 디렉터리에 저장해야 합니다.
2. video_to_image.sh 파일을 실행합니다.
3. save.py 파일을 실행합니다.
