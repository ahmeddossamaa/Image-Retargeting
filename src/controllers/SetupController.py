import base64

from flask import jsonify, request
from flask_socketio import emit

from config.constants import Events
from config.decorators import Decorators


class SetupController:
    def __init__(self):
        pass

    @Decorators.Sockets.On(Events.CONNECT.value)
    def connect(*args):
        print("Connected!")

    @Decorators.Sockets.On(Events.DISCONNECT.value)
    def disconnect(*args):
        pass

    @Decorators.Sockets.On(Events.ERROR.value)
    def error(*args):
        pass

    @staticmethod
    @Decorators.Routers.Post("/test")
    def test():
        if 'image' not in request.files:
            return 'No file found', 400

        file = request.files['image']
        if file.filename == '':
            return 'No selected file', 400

        file_data = file.read()

        # base64_data = base64.b64encode(file_data)

        return file_data
