import cv2
import os
import numpy as np

def extract_frames(video_path, frames_to_extract=30):
    """Extract frames from a video file and save them as images."""
    # Create a unique output directory based on the video filename
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f'extracted_frames_{video_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get total frame count and video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nVideo: {video_path}")
    print(f"Info: {total_frames} frames, {fps} fps, {duration:.2f} seconds")
    
    # Calculate the frame interval
    frame_interval = max(1, total_frames // frames_to_extract)
    
    count = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract every frame_interval frames
        if frame_count % frame_interval == 0 and count < frames_to_extract:
            # Save frame as JPEG file
            filename = os.path.join(output_dir, f"frame_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Extracted: {filename}")
            count += 1
        
        frame_count += 1
        
        # Break if we've extracted enough frames
        if count >= frames_to_extract:
            break
    
    # Release the video capture object
    cap.release()
    
    print(f"Completed! Extracted {count} frames from {video_path}")
    return True

# Dictionary of videos to process with frame counts
videos = {
    'images/gauzey.mp4': 60,
    'images/syringey.mp4': 60,
    'images/bandagey.mp4': 60,
    'images/fieldsand2.mp4': 20,
    'images/fieldsand.mp4': 20,
    'images/fieldburg2.mp4': 20,
    'images/fieldburg.mp4': 20,
    'images/dog3.mp4': 20,
    'images/dog2.mp4': 20,
    'images/dog.mp4': 20,
    'images/backdog.mp4': 20,
    'images/bandage1.mp4': 30,
    'images/bandage2.mp4': 30,
    'images/syringe1.mp4': 30,
    'images/syringe2.mp4': 30,
    'images/gauze.mp4': 60,
    'images/back1.mp4': 40,
    'images/back2.mp4': 40,
    'images/burgsand1.mp4': 30,
    'images/burgsand2.mp4': 30,
    'images/three.mp4': 30,
    'images/three1.mp4': 30,
    'images/sandup.mp4': 30,
    'images/sanddown.mp4': 30,
    'images/burgup.mp4': 30,
    'images/burgdown.mp4': 30,
    'images/nullvid.mp4': 200
}

# Process each video
for video_path, frame_count in videos.items():
    extract_frames(video_path, frames_to_extract=frame_count) 