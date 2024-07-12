# 정리

## 데이터 준비
1. mp4파일을 contensts 디렉터리에 넣는다
2. mp4파일을 20FPS의 이미지로 변환
    ```
    bash video_to_image.sh
    ```
## 도커허브에 도커 이미지 빌드하기
```
bash build_docker.sh
```
## Kubernetes 실행하기
```
bash kube.sh
```

### 이미지 빌드 및 kubernetes실행 셸파일은 메인에 통합될 수 있음

mongo-express: 8081
mongodb: 27017
docker: 8000