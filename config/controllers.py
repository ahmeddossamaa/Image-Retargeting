from src.controllers.SetupController import SetupController

controllers = [
    SetupController
]


def initializeControllers():
    for controller in controllers:
        controller()
