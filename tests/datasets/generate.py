import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from src.processors.Combiner import Combiner
from src.processors.sc.ConnectedSC import ConnectedSC
from src.processors.sc.ForwardSC import ForwardSC
from src.processors.sc.MiddleSC import MiddleSC
from util.Image import Image

path = "../../../Datasets/ReDWeb-S-Modules"

dirs = sorted(list(map(int, os.listdir(path))))

exceptions = []

ratio = 0.75


def backward(i, rgb, energy):
    backward = ConnectedSC(rgb, energy, ratio)()

    Image.save(backward, f"{path}/{i}/backward.jpg")


def middle(i, rgb, energy):
    middle = MiddleSC(rgb, energy, ratio)()

    Image.save(middle, f"{path}/{i}/forward-middle.jpg")


def forward(i, rgb, energy):
    forward = ForwardSC(rgb, energy, ratio)()

    Image.save(forward, f"{path}/{i}/forward.jpg")


exe = ThreadPoolExecutor(max_workers=3)


for i in dirs:
    if i == 1000:
        break

    try:
        dir = f"{path}/{i}"

        img = Image(f"{dir}/rgb.jpg")
        rgb = img.rgb()

        depth = Image(f"{dir}/depth.jpg")()

        energy = Combiner(img, depth=depth, invert=True)().image()

        backward = ConnectedSC(rgb, energy, ratio)()
        Image.save(backward, f"{dir}/u-backward.jpg")

        forward = ForwardSC(rgb, energy, ratio)()
        Image.save(forward, f"{dir}/v-forward.jpg")

        middle = MiddleSC(rgb, energy, ratio)()
        Image.save(middle, f"{dir}/w-forward-middle.jpg")

        # exe.submit(backward, i, rgb, energy)
        # exe.submit(middle, i, rgb, energy)
        # exe.submit(forward, i, rgb, energy)

        if os.path.exists(f"{dir}/result.jpg"):
            os.remove(f"{dir}/result.jpg")

        if os.path.exists(f"{dir}/backward.jpg"):
            os.remove(f"{dir}/backward.jpg")

        if os.path.exists(f"{dir}/forward.jpg"):
            os.remove(f"{dir}/forward.jpg")

        if os.path.exists(f"{dir}/forward-middle.jpg"):
            os.remove(f"{dir}/forward-middle.jpg")
    except Exception as e:
        exceptions.append({
            'message': f"Exception at {i}",
            'e': e
        })

    print("==========================================>", i)

for i in exceptions:
    print(i['message'], i['e'])
