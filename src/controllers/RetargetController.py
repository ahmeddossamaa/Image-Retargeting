import time

from config.decorators import Decorators
from flask import request, jsonify, Response
from config.helper import Helper
from src.processors.sc.ConnectedSC import ConnectedSC
from src.processors.SobelFilter import SobelFilter
from utils.Image import Image


class RetargetController:
    @staticmethod
    @Decorators.Routers.Post("/connected")
    def connected():
        try:
            start_time = time.perf_counter()

            files = request.files

            img = files['image']
            img_mimetype = img.mimetype

            img = Image(img.stream.read(), gray=False, decode=True)

            img_org = img.rgb()
            img_rgb = img.rgb()
            img_gray = img.gray()

            energy = SobelFilter(img_gray)().image()

            img_rgb, energy = ConnectedSC(img_rgb, energy, 0.75)()

            end_time = time.perf_counter()

            # Plotter.images([img_org, img_rgb], 1, 2)

            # Plotter.images([img_org, img_rgb, img_gray], 1, 3)

            total_time = end_time - start_time

            print(total_time)

            stream = Helper.Image.get_stream(img_rgb, mimetype=img_mimetype)

            return Response(stream, mimetype=img_mimetype)
        except Exception as e:
            # print(e)
            return jsonify(e.args)
