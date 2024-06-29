from src.controllers.RetargetController import RetargetController
from src.controllers.SetupController import SetupController
from src.controllers.VideoController import VideoController

controllers = [
    SetupController,
    RetargetController,
    VideoController
]


def initializeControllers():
    for controller in controllers:
        controller()
