import matplotlib.pyplot as plt
from histogram import grayHistogram
import numpy as np
import cv2 as cv
import os

def otsu_threshold(img):
    hist = grayHistogram(img)

    if isinstance(hist, tuple) and len(hist) >= 1:
        counts = np.array(hist[0]).ravel()
    else:
        counts = np.array(hist).ravel()
    counts = counts.astype(np.float64)

    total = counts.sum()
    if total == 0:
        raise ValueError("Histogram total count is zero, cannot normalize.")

    bins = np.arange(counts.size)

    J_theta = None
    theta = None

    for i in range(1, counts.size):
        w0 = counts[:i].sum()
        w1 = counts[i:].sum()

        if w0 == 0 or w1 == 0:
            continue

        p_0 = w0 / total
        p_1 = w1 / total

        mean_0 = (counts[:i] * bins[:i]).sum() / w0
        mean_1 = (counts[i:] * bins[i:]).sum() / w1

        std_0 = np.sqrt((((bins[:i] - mean_0) ** 2) * counts[:i]).sum() / w0)
        std_1 = np.sqrt((((bins[i:] - mean_1) ** 2) * counts[i:]).sum() / w1)

        J = p_0 * (std_0 ** 2) + p_1 * (std_1 ** 2)

        if J_theta is None or J < J_theta:
            J_theta = J
            theta = i

    if theta is None:
        raise ValueError("Could not determine optimal threshold.")
    return theta, counts

if __name__ == "__main__":
    root = os.getcwd()
    imgPath = os.path.join(root, 'rice.jpeg')
    if not os.path.exists(imgPath):
        print(f"Error: image not found at {imgPath}")

    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image")
    
    theta, counts = otsu_threshold(img)
    plt.plot(counts)
    plt.axvline(x=theta, color='r', linestyle='--', label=f'Theta = {theta}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()