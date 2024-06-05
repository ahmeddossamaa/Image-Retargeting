import numpy as np
from cv2 import VideoCapture, VideoWriter, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, cvtColor, \
    COLOR_BGR2GRAY

from config.constants import DataPath
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from src.processors.sc.MiddleSC import MiddleSC
from utils.Image import Image

cat = "ball"
ratio = 0.75

def extract_frames(video_path):
    cap = VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame
        Image.save(frame, f'frames/{cat}/{frame_count}.jpg')

        frame_count += 1

    cap.release()


def drop_some_frames(input_video_path, output_video_path, skip_every_n_frame=10):
    cap = VideoCapture(input_video_path)
    fourcc = VideoWriter().fourcc(*'mp4v')
    fps = int(cap.get(CAP_PROP_FPS))
    width = int(cap.get(CAP_PROP_FRAME_WIDTH) * ratio)
    height = int(cap.get(CAP_PROP_FRAME_HEIGHT))

    out = VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        print(frame_count)

        if frame_count % skip_every_n_frame == 0:
            continue

        gray = cvtColor(frame, COLOR_BGR2GRAY)

        energy = SobelFilter(gray)().image()

        result = MiddleSC(frame, energy, 0.75)()

        out.write(result)

        if frame_count == 25:
            break

    cap.release()
    out.release()


# extract_frames(f"{DataPath.INPUT_PATH.value}/videos/{cat}.mp4")

drop_some_frames(
    f"{DataPath.INPUT_PATH.value}/videos/{cat}.mp4",
    f"{DataPath.INPUT_PATH.value}/videos/{cat}_retargeted.mp4",
    skip_every_n_frame=3
)
