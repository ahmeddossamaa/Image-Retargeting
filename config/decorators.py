import time


class Decorators:
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
