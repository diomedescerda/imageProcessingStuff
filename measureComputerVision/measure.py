import os, sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

    return mm_per_pixel, best_contour

def fill_holes(binary_img, max_gap_pixels=100, line_thickness=3):
    kernel_length = max_gap_pixels
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_length, line_thickness))
    
    opened = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)
    
    return opened

def filter_contours_by_area(contours, min_area=100, max_area=12526455):
    filtered = []
    for contour in contours:
        area = cv.contourArea(contour)
        if min_area < area < max_area:
            filtered.append(contour)
    return filtered

def measure_with_pca(contour, mm_per_pixel):
    points = contour.reshape(-1, 2).astype(np.float64)
    
    pca = PCA()
    projected_points = pca.fit_transform(points)
    
    min_proj_x = np.min(projected_points[:, 0])
    max_proj_x = np.max(projected_points[:, 0])
    min_proj_y = np.min(projected_points[:, 1])
    max_proj_y = np.max(projected_points[:, 1])
    
    length_pixels = max_proj_x - min_proj_x
    width_pixels = max_proj_y - min_proj_y
    
    length_mm = length_pixels * mm_per_pixel
    width_mm = width_pixels * mm_per_pixel
    
    mean = pca.mean_
    eigenvectors = pca.components_
    
    return length_mm, width_mm, eigenvectors, mean

def draw_pca_measurements(image, contour, mm_per_pixel):
    points = contour.reshape(-1, 2).astype(np.float64)

    pca = PCA(n_components=2)
    pca.fit(points)
    projected = pca.transform(points)

    min_x, max_x = projected[:, 0].min(), projected[:, 0].max()
    min_y, max_y = projected[:, 1].min(), projected[:, 1].max()

    corners_pca = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])

    corners_img = pca.inverse_transform(corners_pca).astype(int)
    cv.polylines(image, [corners_img], True, (0, 255, 0), 2)

    length_pixels = max_x - min_x
    width_pixels = max_y - min_y
    length_mm = length_pixels * mm_per_pixel
    width_mm = width_pixels * mm_per_pixel

    center_pca = np.array([(min_x + max_x)/2, (min_y + max_y)/2])
    center_img = pca.inverse_transform(center_pca.reshape(1, -1))[0].astype(int)

    axis_scale = 50
    v1 = pca.components_[0] * axis_scale
    v2 = pca.components_[1] * axis_scale

    cv.arrowedLine(image, tuple(center_img), tuple((center_img + v1).astype(int)), (0, 0, 255), 3)  # Length axis
    cv.arrowedLine(image, tuple(center_img), tuple((center_img + v2).astype(int)), (255, 0, 0), 3)  # Width axis

    cv.putText(image, f"L:{length_mm:.1f}mm", (center_img[0]+10, center_img[1]-20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.putText(image, f"W:{width_mm:.1f}mm", (center_img[0]+10, center_img[1]+20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return length_mm, width_mm

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
    theta = otsu_threshold(smoothed)[0]
    _, binary = cv.threshold(smoothed, theta, 255, cv.THRESH_BINARY)
    binary = cv.bitwise_not(binary)
    kernel = np.ones((5,5),dtype=np.uint8)
    dilated = cv.dilate(binary, kernel, iterations=2)
    eroded = cv.erode(dilated, kernel, iterations=4)
    filled = fill_holes(eroded)

    contours = cv.findContours(filled, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        print("No contours found")
        return

    mm_per_pixel, coin_contour = get_mm_per_pixel(img, contours, ref_size_mm)
    print(f"Millimeters per pixel: {mm_per_pixel}")

    display_img = img.copy()
    results = []

    filtered = filter_contours_by_area(contours)

    print(f"Total contours: {len(filtered)}")
    for contour in filtered:
        if np.array_equal(contour, coin_contour):
            continue

        length_mm, width_mm, eigenvectors, mean = measure_with_pca(contour, mm_per_pixel)
        draw_pca_measurements(display_img, contour, mm_per_pixel)
        results.append({'length_mm': length_mm, 'width_mm': width_mm})

    plt.figure(figsize=(12, 8))
    plt.imshow(cv.cvtColor(display_img, cv.COLOR_BGR2RGB))
    plt.title("PCA-Based Object Measurement", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    measure_main("easy.jpg")