import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def houghLineTransform(imgName):
    root = os.getcwd()
    imgPath = os.path.join(root, imgName)
    if not os.path.exists(imgPath):
        print(f"Error: image not found at {imgPath}")
        return

    # load as grayscale
    img_gray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Error: Could not load image")
        return

    cannyEdges = cv.Canny(img_gray, 50, 180)

    plt.figure()
    # show grayscale and edges with proper cmap
    plt.subplot(131)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(cannyEdges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')

    disResolution = 1
    angleResolution = np.pi / 180
    threshold = 21
    lines = cv.HoughLines(cannyEdges, disResolution, angleResolution, threshold)

    img_color = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

    if lines is None:
        print("No lines detected.")
    else:
        k = 3000
        line_thickness = 1
        line_color = (0, 0, 255)
        for curLine in lines:
            rho, theta = curLine[0]
            dhat = np.array([np.cos(theta), np.sin(theta)])
            d = rho * dhat
            lhat = np.array([-dhat[1], dhat[0]])
            p1 = (d + k * lhat).astype(np.int32)
            p2 = (d - k * lhat).astype(np.int32)
            cv.line(img_color, tuple(p1), tuple(p2), line_color, line_thickness, lineType=cv.LINE_AA)

    plt.subplot(133)
    plt.imshow(cv.cvtColor(img_color, cv.COLOR_BGR2RGB))
    plt.title('Hough Lines')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    houghLineTransform('rice.jpeg')