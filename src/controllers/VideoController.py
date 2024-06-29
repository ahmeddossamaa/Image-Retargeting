from time import sleep

# from flask import request, jsonify

from config.decorators import Decorators
from src.processors.Combiner import Combiner
from src.processors.sc.MiddleSC import MiddleSC
from utils.Image import Image
from utils.Pipeline import Pipeline
from utils.Socket import Socket
from utils.Worker import Worker


def run(num):
    print(f"Start process: {num}")
    sleep(1)
    print(f"Finish process: {num + 1}")

    return num + 1


def emitter(num):
    print("from emitter")
    Socket.get_instance().emit("video", num)


class VideoController:
    pipeline = None

    @staticmethod
    # @Decorators.Routers.Post("/video/test")
    @Decorators.Sockets.On("video")
    def video(num):
        # num = int(request.form.get('num'))
        num = int(num)

        if VideoController.pipeline is None:
            VideoController.pipeline = Pipeline([
                Worker.make(1, run, 2),
                Worker.make(2, run, 3),
                Worker.make(3, emitter)
            ])

        VideoController.pipeline.push(num)

        # return jsonify(VideoController.pipeline.get())
