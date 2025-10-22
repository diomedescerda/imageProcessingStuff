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

    img_gray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Error: Could not load image")
        return

    cannyEdges = cv.Canny(img_gray, 50, 180)

    plt.figure()
    plt.subplot(131)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(cannyEdges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')
    lines = myHoughLines(cannyEdges)


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

def myHoughLines(edge_img, rho_res=1, theta_res=np.pi/180, threshold=22):
    y_idxs, x_idxs = np.nonzero(edge_img)
    height, width = edge_img.shape

    diag_len = int(np.ceil(np.hypot(width, height)))

    thetas = np.arange(0, np.pi, theta_res)
    rhos = np.arange(-diag_len, diag_len + 1, rho_res)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        rhos_values = x * cos_t + y * sin_t
        rho_indices = np.round((rhos_values - rhos[0]) / rho_res).astype(int)
        accumulator[rho_indices, np.arange(len(thetas))] += 1

    lines = []
    for r_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            if accumulator[r_idx, t_idx] >= threshold:
                rho = rhos[r_idx]
                theta = thetas[t_idx]
                lines.append([[rho, theta]])

    return lines

if __name__ == '__main__':
    houghLineTransform('rice.jpeg')