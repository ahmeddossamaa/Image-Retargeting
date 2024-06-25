from time import sleep

from utils.Worker import Worker


def run(num):
    print(f"start process: {num}")
    sleep(1)
    print(f"finish process: {num}")

    return num + 1


queues = {}
workers = []
for i in range(0, 3):
    old = workers[i - 1] if len(workers) > 0 else None

    new = Worker(3 - i + 1, run, queues, 3 - i)

    workers.append(new)
    # queues.append(new.args_queue)

print("started")

workers[0].put(1)
# print(workers[0].thread.is_alive())
