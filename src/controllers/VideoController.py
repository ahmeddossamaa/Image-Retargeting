import numpy as np
from cv2 import VideoWriter, VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
from config.decorators import Decorators
from src.processors.Fucker import Fucker
from src.processors.sc.ImprovedSC import ImprovedSC
from src.processors.sc.MiddleSCI import MiddleSCI
from util.Image import Image
from util.Pipeline import Pipeline
from util.Socket import Socket
from util.Worker import Worker

global frame_counter, frames_count


def convert(obj: MiddleSCI, out: VideoWriter, *args):
    print("converted", frame_counter)
    return obj, out, obj.convert(*args)


def carve(obj: MiddleSCI, out: VideoWriter, *args):
    print("accumulated", frame_counter)
    args = obj.accumulate(*args)
    res = obj.remove(args)
    return out, res


def save(out_path):
    def saver(out: VideoWriter, *args):
        img = np.array(args)[0]
        out.write(img)
        global frame_counter, frames_count
        frame_counter += 1
        Socket.get_instance().emit("frame", frame_counter / frames_count)
        if frame_counter == frames_count:
            print("released")
            out.release()
            Socket.get_instance().emit("video", out_path)

    return saver


class VideoController:
    pipeline = None

    @staticmethod
    @Decorators.Sockets.On("video")
    def video(path, out_path, ratio):
        if VideoController.pipeline is None:
            VideoController.pipeline = Pipeline([
                Worker.make(1, convert, 2),
                Worker.make(2, carve, 3),
                Worker.make(3, save(out_path)),
            ])

        ratio = float(ratio)

        source = VideoCapture(path)
        fps = int(source.get(CAP_PROP_FPS))
        width = int(source.get(CAP_PROP_FRAME_WIDTH) * ratio)
        height = int(source.get(CAP_PROP_FRAME_HEIGHT))

        fourcc = VideoWriter().fourcc(*'mp4v')
        target = VideoWriter(out_path, fourcc, fps, (width, height))

        global frame_counter, frames_count
        frame_counter = 0
        frames_count = 0

        while source.isOpened():
            ret, frame = source.read()
            if not ret:
                break

            frames_count += 1
            img = Image("", data=frame)
            sc = ImprovedSC(img, ratio, converter=Fucker, prev_matrix=True)
            VideoController.pipeline.push((sc, target, img))

        source.release()
