# content1 png파일로 변환
ffmpeg -i ./contents/1/*.mp4 -vf "fps=20" -vsync vfr -q:v 2 ./contents/1/frame_%04d.png
# content2 png파일로 변환
ffmpeg -i ./contents/2/*.mp4 -vf "fps=20" -vsync vfr -q:v 2 ./contents/2/frame_%04d.png
# content3 png파일로 변환
ffmpeg -i ./contents/3/*.mp4 -vf "fps=20" -vsync vfr -q:v 2 ./contents/3/frame_%04d.png