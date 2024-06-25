import os
import re
import shutil


def transfer():
    PATH = "../../../Datasets/ReDWeb-S/trainset"

    for i in os.listdir(PATH):
        for j in os.listdir(f"{PATH}/{i}"):
            name = re.sub(r'\.jpg|\.png', '', j)

            dir_name = f"../../../Datasets/ReDWeb-S-Modules/{name}"

            if not os.path.exists(dir_name):
                os.mkdir(dir_name, )
                print("created", dir_name)

            shutil.copy2(f"{PATH}/{i}/{j}", f"{dir_name}/{i.lower()}.jpg")


def rename():
    PATH = "../../../Datasets/ReDWeb-S-Modules"

    c = 1
    for i in os.listdir(PATH):
        os.rename(f"{PATH}/{i}", f"{PATH}/{c}")

        c += 1


# transfer()
# rename()
