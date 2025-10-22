import os, sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsu import otsu_threshold

def apply_smoothing(gray, method="gaussian", ksize=5):
    if ksize % 2 == 0:
        ksize += 1
    if method == "gaussian":
        return cv.GaussianBlur(gray, (ksize, ksize), 0)
    elif method == "median":
        return cv.medianBlur(gray, ksize)
    elif method == "bilateral":
        return cv.bilateralFilter(gray, d=ksize, sigmaColor=75, sigmaSpace=75)
    return cv.GaussianBlur(gray, (ksize, ksize), 0)

def get_mm_per_pixel(img, contours, ref_size_mm=23.7):
    display_img = img.copy()
    cv.drawContours(display_img, contours, -1, (0,255,0), 3)

    circularity = 0
    best_contour = None
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        
        if perimeter > 300:
            temp = (4 * np.pi * area) / (perimeter ** 2)
            if temp > circularity:
                circularity = temp
                best_contour = contour

    area = cv.contourArea(best_contour)
    diameter_pixels = 2 * np.sqrt(area / np.pi)
    mm_per_pixel = ref_size_mm / diameter_pixels

    return mm_per_pixel

def measure_main(imgName, ref_size_mm=23.7):
    root = os.getcwd()
    imgPath = os.path.join(root, imgName)
    if not os.path.exists(imgPath):
        print(f"Error: image not found at {imgPath}")
        return
    img = cv.imread(imgPath)
    if img is None:
        print("Error: Could not load image")
        return

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    smoothed = apply_smoothing(gray, method="gaussian", ksize=5)
    theta = otsu_threshold(smoothed)[0] - 10
    _, binary = cv.threshold(smoothed, theta, 255, cv.THRESH_BINARY)
    binary = cv.bitwise_not(binary)
    kernel = np.ones((5,5),dtype=np.uint8)
    eroded = cv.erode(binary, kernel, iterations=3)
    dilated = cv.dilate(eroded, kernel, iterations=2)

    contours = cv.findContours(dilated, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        print("No contours found")
        return

    mm_per_pixel = get_mm_per_pixel(img, contours, ref_size_mm)
    print(f"Millimeters per pixel: {mm_per_pixel}")
    

if __name__ == "__main__":
    measure_main("example.jpeg")