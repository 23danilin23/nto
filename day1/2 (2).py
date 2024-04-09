import cv2
import os

def extract_frames(video_path, output_folder):
    # Check if the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Extract frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frame to the output folder
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()
    print(f"{frame_count} frames extracted.")

if __name__ == "__main__":
    # Path to the video file
    video_path = input("specify video file name\n") + ".mp4"
    
    # Folder to save frames
    output_folder = "./" + input("specify folder to save to\n")

    # Extract frames
    extract_frames(video_path, output_folder)
