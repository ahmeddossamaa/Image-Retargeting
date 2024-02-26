import time

from utils.App import App
from utils.Socket import Socket


class Decorators:
    class Sockets:
        @staticmethod
        def On(event):
            return Socket.get_instance().on(event)

    class Routers:
        @staticmethod
        def Get(rule):
            return App.get_instance().get(rule)

        @staticmethod
        def Post(route):
            return App.get_instance().post(route)

        @staticmethod
        def Put(route):
            return App.get_instance().put(route)

        @staticmethod
        def Delete(route):
            return App.get_instance().delete(route)

    class Loggers:
        @staticmethod
        def log_class_method_time(func):
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()

                execution_time = end_time - start_time

                class_name = args[0].__class__.__name__

                print(f"Function '{class_name}.{func.__name__}' executed in {execution_time:.4f} seconds")

                return result

            return wrapper
