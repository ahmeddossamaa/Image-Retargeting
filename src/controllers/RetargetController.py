import time

import numpy as np
from matplotlib import pyplot as plt

from config.decorators import Decorators
from flask import request, jsonify, Response
from config.helper import Helper
from config.plotter import Plotter
from src.processors.CannyProcessor import CannyProcessor
from src.processors.sc.ConnectedSC import ConnectedSC
from src.processors.SobelFilter import SobelFilter
from utils.Image import Image
from PIL import Image
import io


class RetargetController:
    @staticmethod
    @Decorators.Routers.Post("/connected")
    def connected():
        try:
            print("hello from connected")
            start_time = time.perf_counter()

            img_file = request.files['image']
            if img_file is None:
                return jsonify({'error': 'No image file provided'}), 400

            img_content = img_file.read()

            img = Image.open(io.BytesIO(img_content))

            print(f"File name: {img_file.filename}")

            img_rgb = img.convert('RGB')
            img_gray = img.convert('L')

            ratio = float(request.form.get('ratio', 0.5))

            energy = SobelFilter(img_gray)().image()
            processed_img = ConnectedSC(
                np.array(img_rgb), energy, ratio, color=False)()

            Image.fromarray(processed_img.astype('uint8')).show()

            end_time = time.perf_counter()
            print("Processing time:", end_time - start_time)

            img_mimetype = 'image/png'
            stream = Helper.Image.get_stream(processed_img, img_mimetype)

            return Response(stream, mimetype=img_mimetype)

        except Exception as e:
            print("Error:", e)
            return jsonify({'error': str(e)}), 500

    @staticmethod
    @Decorators.Routers.Post("/energy")
    def energy():
        try:
            files = request.files

            img = files['image']
            mimetype = img.mimetype

            img = Image(img.stream.read(), gray=False, decode=True)

            img_gray = img.gray()

            energy = SobelFilter(img_gray)().image()

            Plotter.image(energy)

            stream = Helper.Image.get_stream(img_gray, mimetype)
            print("hello")

            return Response(stream, mimetype=mimetype)
        except Exception as e:
            return jsonify(e.args)
