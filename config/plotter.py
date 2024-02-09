import matplotlib.pyplot as plt


class Plotter:
    @staticmethod
    def image(image):
        plt.imshow(image)
        plt.show()

    @staticmethod
    def images(images, rows, columns, size=(10, 7)):
        figure = plt.figure(figsize=size)

        for i, img in enumerate(images):
            figure.add_subplot(rows, columns, i + 1)

            plt.imshow(img)

        plt.show()
