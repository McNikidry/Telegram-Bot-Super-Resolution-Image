import numpy as np
import cv2
import matplotlib.pyplot as plt

class DetectionFigures():
    def __init__(self):
        self.__red = (255, 0, 0)
        self.__green = (0, 255, 0)
        self.__blue = (0, 0, 255)

    def predict(self, image: np.array) -> np.array:
        img = np.copy(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            approx = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True)

            # using drawContours() function
            if len(approx) == 3:
                cv2.drawContours(img, [contour], 0, self.__red, -1)
            elif len(approx) == 4:
                cv2.drawContours(img, [contour], 0, self.__blue, -1)
            else:
                cv2.drawContours(img, [contour], 0, self.__green, -1)

        return img


if __name__ == '__main__':
    img = cv2.imread('test_figures.jpg')
    detectionfigures = DetectionFigures()
    img_result = detectionfigures.predict(img)
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(img)
    axs[0].set_title('Original image')
    axs[1].imshow(img_result)
    axs[1].set_title('Predicted image')
    plt.show()