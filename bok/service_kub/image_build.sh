#!/bin/bash

# Docker Hub 사용자 이름
DOCKER_USER=joobok

# 이미지 이름과 태그
IMAGE_NAME=mongodb
IMAGE_TAG=v1.0

# Docker Hub 로그인
docker login
# Docker 이미지를 빌드
docker build -t ${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG} .
# 이미지 푸시
docker push ${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}