# contents 디렉터리의 모든 mp4파일을 20FPS로 png파일로 변환
ffmpeg -i ./contents/*.mp4 -vf "fps=20" -vsync vfr -q:v 2 ./contents/frame_%04d.png