from src.controllers.RetargetController import RetargetController
from src.controllers.SetupController import SetupController

controllers = [
    SetupController,
    RetargetController
]


def initializeControllers():
    for controller in controllers:
        controller()
