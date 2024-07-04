# mp4파일을 1FPS로 변환
sudo apt install ffmpeg
ffmpeg -i video1.mp4 -vf "select='not(mod(n,30))',setpts='N/(30*TB)'" -vsync vfr -q:v 2 output_frame_%04d.png
