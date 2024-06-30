from flask import Flask

class App:
    _app: Flask = None

    def __init__(self, name):
        if App._app is not None:
            raise Exception("App already created!")
        App._app = Flask(name)

    @staticmethod
    def create_instance(name):
        App(name)
        return App.get_instance()

    @staticmethod
    def get_instance():
        return App._app

def initializeApp(name):
    return App.create_instance(name)