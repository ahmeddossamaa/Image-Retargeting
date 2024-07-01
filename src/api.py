import numpy as np
from cv2 import VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, VideoWriter

from config.constants import DataPath
from config.plotter import Plotter
from src.processors.Combiner import Combiner
from src.processors.Fucker import Fucker
from src.processors.sc.MiddleSCI import MiddleSCI
# from util.Image import Image
# from util.Pipeline import Pipeline
# from util.Worker import Worker
from util.Image import Image
from util.Pipeline import Pipeline
from util.Worker import Worker


def retarget_image(path: str, ratio: float):
    img = Image(path)

    return MiddleSCI(img, ratio, converter=Fucker)()


def retarget_video(path: str, out_path: str, ratio: float):
    source = VideoCapture(path)

    fps = int(source.get(CAP_PROP_FPS))
    width = int(source.get(CAP_PROP_FRAME_WIDTH) * ratio)
    height = int(source.get(CAP_PROP_FRAME_HEIGHT))

    fourcc = VideoWriter().fourcc(*'mp4v')
    target = VideoWriter(out_path, fourcc, fps, (width, height))

    global frame_count

    count = 0
    frame_count = 0

    def convert(obj: MiddleSCI, out: VideoWriter, *args):
        return obj, out, obj.convert(*args)

    def carve(obj: MiddleSCI, out: VideoWriter, *args):
        args = obj.accumulate(*args)
        res = obj.remove(args)

        return out, res

    def save(out: VideoWriter, *args):
        global frame_count

        frame_count += 1

        image = np.array(args)[0]

        out.write(image)
        if frame_count == count:
            out.release()

        print("saved", frame_count)

    pipeline = Pipeline([
        Worker.make(1, convert, 2),
        Worker.make(2, carve, 3),
        Worker.make(3, save)
    ])

    while source.isOpened():
        ret, frame = source.read()
        if not ret:
            break

        count += 1
        # if count % 3 == 0:
        #     continue

        img = Image("", data=frame)

        sc = MiddleSCI(img, ratio, converter=Combiner, prev_matrix=True)

        pipeline.push((sc, target, img))

        if count == 104:
            break

    source.release()


# image = retarget_image("../data/input/moon.jpg", 0.75)
#
# Plotter.image(image)

# retarget_video(f"{DataPath.INPUT_PATH.value}/videos/ball.mp4", f"{DataPath.INPUT_PATH.value}/videos/ball_retarget.mp4", ratio=0.75)
