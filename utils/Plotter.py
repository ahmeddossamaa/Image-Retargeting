import matplotlib.pyplot as plt


class Plotter:
    @staticmethod
    def image(image):
        plt.imshow(image)
        plt.show()
