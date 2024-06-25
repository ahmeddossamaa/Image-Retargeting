from queue import Queue
from threading import Thread


class Worker:
    def __init__(self, worker_id, func, queues: dict, next_id):
        self.worker_id = worker_id
        self.func = func
        self.next_id = next_id

        self.args_queue = Queue()

        queues[f"{worker_id}"] = self.args_queue

        self.is_running = True

        self.thread = Thread(target=self.run, args=(queues, ))
        # self.thread.daemon = True
        self.thread.start()

    def run(self, queues):
        try:
            while self.is_running:
                if self.args_queue.empty():
                    continue

                args = self.args_queue.get()

                res = self.func(args)

                if self.next_id is not None:
                    print("next_id", self.next_id)
                    q = queues.get(f"{self.next_id}")

                    q.put(res)

                self.args_queue.task_done()
        except Exception as e:
            print(e, f"from {self.worker_id}")

    def put(self, task):
        self.args_queue.put(task)

    def stop(self):
        self.is_running = False
        self.thread.join()

    @staticmethod
    def start_all():
        pass
