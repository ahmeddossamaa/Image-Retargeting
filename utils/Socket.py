from flask import Flask
from flask_socketio import SocketIO


class Socket:
    _socket: SocketIO = None

    def __init__(self, app):
        if Socket._socket is not None:
            raise Exception("Socket already created!")

        Socket._socket = SocketIO(app)

    @staticmethod
    def create_instance(app: Flask):
        Socket(app)

        return Socket.get_instance()

    @staticmethod
    def get_instance():
        return Socket._socket


def initializeSocket(app: Flask):
    return Socket.create_instance(app)
