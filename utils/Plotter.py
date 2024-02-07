import matplotlib.pyplot as plt


class Plotter:
    @staticmethod
    def image(title, image):
        plt.imshow(image)
        plt.title(title)
        plt.show()

