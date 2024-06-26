class Main:
    __app = None

    def __init__(self):
        if Main.__app is not None:
            raise Exception("App is already running!")

        from util.App import initializeApp
        self.__app = initializeApp(__name__)

        from util.Socket import initializeSocket
        initializeSocket(self.__app)

        from config.controllers import initializeControllers
        initializeControllers()

    @staticmethod
    def get_instance():
        return Main()

    def run(self):
        self.__app.run()


if __name__ == "__main__":
    Main.get_instance().run()
