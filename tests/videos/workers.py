from time import sleep

from utils.Pipeline import Pipeline
from utils.Worker import Worker


def run(num):
    print(f"Start process: {num}")
    sleep(1)
    print(f"Finish process: {num + 1}")

    return num + 1


# queues = {}

# first = Worker(1, run, queues, 2)
# second = Worker(2, run, queues, 3)
# third = Worker(3, run, queues)
#
# print("started")
#
# first.put(1)

Pipeline([
    {
        "id": 1,
        "func": run,
        "next_id": 2
    },
    {
        "id": 2,
        "func": run,
        "next_id": 3
    },
    {
        "id": 3,
        "func": run
    },
]).push(1)
