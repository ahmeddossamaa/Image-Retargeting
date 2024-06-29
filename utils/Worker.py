from queue import Queue
from threading import Thread


class Worker:
    def __init__(self, worker_id, func, queues, next_id=None):
        self.worker_id = worker_id
        self.func = func
        self.next_id = next_id
        self.args_queue = Queue()
        self.is_running = True
        self.last_result = None

        # Add this worker's queue to the shared queues dictionary
        queues[worker_id] = self.args_queue

        # Start the worker thread
        self.thread = Thread(target=self.run, args=(queues,))
        self.thread.start()

    def run(self, queues):
        try:
            while self.is_running:
                # Wait for a task from the queue
                args = self.args_queue.get()

                # Process the task
                result = self.func(*args)

                # Store the last result
                self.last_result = result

                # If there is a next worker, pass the result to its queue
                if self.next_id is not None:
                    next_queue = queues.get(self.next_id)
                    if next_queue:
                        next_queue.put(result)

                    if self.args_queue.qsize() == 0:
                        print(f"Worker {self.worker_id} finished")

                # Mark the task as done
                self.args_queue.task_done()
        except Exception as e:
            print(f"Error in worker {self.worker_id}: {e}")

    def put(self, task):
        # Add a task to the queue
        self.args_queue.put(task)

    def get_last(self):
        # Get the last processed result
        return self.last_result

    def stop(self):
        # Stop the worker thread
        self.is_running = False
        self.thread.join()

    @staticmethod
    def make(worker_id, func, next_id=None):
        # Static method to create a worker configuration dictionary
        return {
            "id": worker_id,
            "func": func,
            "next_id": next_id
        }
