#!/bin/bash

# ffmpeg 설치
sudo apt install -y ffmpeg

# 디렉터리 이름을 받을 인자 설정
dir_name=$1

# 컨텐츠 영상을 20FPS로 png파일로 변환
ffmpeg -i ./contents/${dir_name}/*.mp4 -vf "fps=20" -vsync vfr -q:v 2 ./contents/${dir_name}/frame_%04d.png