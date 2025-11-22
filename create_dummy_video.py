import cv2
import numpy as np

def create_dummy_video(filename="dummy_video.mp4", duration=5, fps=25, width=1280, height=720):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    frames = duration * fps
    for i in range(frames):
        # Create a frame with some moving content
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw a moving rectangle
        x = int((i / frames) * (width - 100))
        cv2.rectangle(img, (x, 100), (x + 100, 200), (0, 255, 0), -1)

        # Draw frame number
        cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(img)

    out.release()
    print(f"Created {filename}")

if __name__ == "__main__":
    create_dummy_video()
