import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def grayHistogram():
    root = os.getcwd()
    imgPath = os.path.join(root, 'rice.jpeg')
    if not os.path.exists(imgPath):
        print(f"Error: image not found at {imgPath}")
        return

    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image")
        return

    # plt.figure()
    # plt.imshow(img, cmap='gray')

    hist = cv.calcHist([img], [0], None, [256], [0,256])
    return hist

if __name__ == '__main__':
    hist = grayHistogram()
    plt.figure()
    plt.plot(hist)
    plt.xlabel('bins')
    plt.ylabel('# of pixels')
    plt.show()
