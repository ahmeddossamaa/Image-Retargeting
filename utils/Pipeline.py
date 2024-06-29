from utils.Worker import Worker


class Pipeline:
    current_id = 1
    workers = []
    queues = {}

    def __init__(self, workers):
        for i in workers:
            id = i.get("id")
            func = i.get("func")
            next_id = i.get("next_id") or None

            worker = Worker(id, func, self.queues, next_id)

            try:
                self.workers.append(worker)

                self.current_id += 1
            except Exception as e:
                worker.stop()

                print(f"error at worker {id}", e)

    def push(self, init):
        self.workers[0].put(init)
        return self

    def get(self):
        return self.workers[len(self.workers) - 1].get_last()

    def stop(self):
        for i in self.workers:
            i.stop()
