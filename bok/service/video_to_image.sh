# ffmpeg 설치
sudo apt install -y ffmpeg

# # contents1 영상을 20FPS로 png파일로 변환
# ffmpeg -i ./contents/contents1/*.mp4 -vf "fps=20" -vsync vfr -q:v 2 ./contents/contents1/frame_%04d.png
# # contents2 영상을 20FPS로 png파일로 변환
# ffmpeg -i ./contents/contents2/*.mp4 -vf "fps=20" -vsync vfr -q:v 2 ./contents/contents2/frame_%04d.png
# # contents3 영상을 20FPS로 png파일로 변환
# ffmpeg -i ./contents/contents3/*.mp4 -vf "fps=20" -vsync vfr -q:v 2 ./contents/contents3/frame_%04d.png

# contents 디렉터리의 모든 하위 디렉터디에 있는 mp4파일을 20FPS로 png파일로 변환
find ./contents -type f -name "*.mp4" | while read -r file; do
    ffmpeg -i "$file" -vf "fps=20" -vsync vfr -q:v 2 "$(dirname "$file")/frame_$(basename "$file" .mp4)_%04d.png"
done
