import matplotlib.pyplot as plt


class Plotter:
    @staticmethod
    def __transform_to_rgb(image):
        if len(image.shape) < 3:
            return image

        im2 = image.copy()
        im2[:, :, 0] = image[:, :, 2]
        im2[:, :, 2] = image[:, :, 0]
        return im2

    @staticmethod
    def __set_plt(img, title=""):
        plt.imshow(
            Plotter.__transform_to_rgb(img)
        )
        plt.title(title)

    @staticmethod
    def image(img, title=""):
        Plotter.__set_plt(img, title)

        plt.show()

    @staticmethod
    def images(images, rows, columns, size=(10, 7)):
        figure = plt.figure(figsize=size)

        for i, img in enumerate(images):
            figure.add_subplot(rows, columns, i + 1)

            plt.imshow(
                Plotter.__transform_to_rgb(img)
            )

        plt.show()
