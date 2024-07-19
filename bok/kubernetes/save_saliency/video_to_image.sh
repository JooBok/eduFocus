# ffmpeg 설치
sudo apt install -y ffmpeg

# # contents1 영상을 20FPS로 png파일로 변환
ffmpeg -i ./contents/contents2/*.mp4 -vf "fps=20" -vsync vfr -q:v 2 ./contents/contents2/frame_%04d.png
# # # contents2 영상을 20FPS로 png파일로 변환
# ffmpeg -i ./contents/contents2/*.mp4 -vf "fps=20" -vsync vfr -q:v 2 ./contents/contents2/frame_%04d.png
# # # contents3 영상을 20FPS로 png파일로 변환
# ffmpeg -i ./contents/contents3/*.mp4 -vf "fps=20" -vsync vfr -q:v 2 ./contents/contents3/frame_%04d.png