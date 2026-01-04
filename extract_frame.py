import cv2
import sys
import os

def extract_frame(video_path, output_path, frame_num=30):
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Saved frame {frame_num} to {output_path}")
    else:
        print("Failed to capture frame")
    cap.release()

if __name__ == "__main__":
    extract_frame('assets/walking1.mp4', 'dataset/TestPerson/reference.jpg')
