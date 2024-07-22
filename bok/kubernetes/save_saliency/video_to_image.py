import os
import cv2
import numpy as np

def extract_frames(video_path, frame_rate, max_frames):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    
    output_dir = os.path.dirname(video_path)
    
    count = 0
    extracted_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or extracted_frames >= max_frames:
            break
        
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f'frame_{extracted_frames:04d}.png')
            cv2.imwrite(frame_filename, frame)
            extracted_frames += 1
        
        count += 1
    
    cap.release()

def get_min_frames(video_paths, frame_rate):
    min_frames = float('inf')
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        total_possible_frames = int(total_frames / fps * frame_rate)
        if total_possible_frames < min_frames:
            min_frames = total_possible_frames
    
    return min_frames

def main():
    directories = ['test_user/user2/contents1', 'contents/contents1']  # 이미지로 변경할 디렉터리들
    frame_rate = 20
    all_video_paths = []
    
    for directory in directories:
        video_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]
        all_video_paths.extend(video_paths)
    
    if not all_video_paths:
        print("No video files found in the specified directories.")
        return
    
    max_frames = get_min_frames(all_video_paths, frame_rate)
    
    for video_path in all_video_paths:
        extract_frames(video_path, frame_rate, max_frames)

if __name__ == "__main__":
    main()
