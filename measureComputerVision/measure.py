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

def filter_contours_by_area(contours, min_area=1000, max_area=50000):
    filtered = []
    for contour in contours:
        area = cv.contourArea(contour)
        if min_area < area < max_area:
            filtered.append(contour)
    return filtered

def measure_with_pca(contour, mm_per_pixel):
    # Reshape contour to point cloud
    points = contour.reshape(-1, 2).astype(np.float64)
    
    # Apply sklearn PCA
    pca = PCA()
    projected_points = pca.fit_transform(points)
    
    # Get range along each principal component
    min_proj_x = np.min(projected_points[:, 0])
    max_proj_x = np.max(projected_points[:, 0])
    min_proj_y = np.min(projected_points[:, 1])
    max_proj_y = np.max(projected_points[:, 1])
    
    # Calculate dimensions in pixels
    length_pixels = max_proj_x - min_proj_x
    width_pixels = max_proj_y - min_proj_y
    
    # Convert to mm
    length_mm = length_pixels * mm_per_pixel
    width_mm = width_pixels * mm_per_pixel
    
    # Get components for visualization
    mean = pca.mean_
    eigenvectors = pca.components_
    
    return length_mm, width_mm, eigenvectors, mean

def draw_pca_measurements(image, contour, mm_per_pixel):
    # Convert contour to points
    points = contour.reshape(-1, 2).astype(np.float64)
    
    # Apply PCA
    pca = PCA()
    projected_points = pca.fit_transform(points)
    
    # Get dimensions in PCA space
    min_px, max_px = np.min(projected_points[:, 0]), np.max(projected_points[:, 0])
    min_py, max_py = np.min(projected_points[:, 1]), np.max(projected_points[:, 1])
    
    length_pixels = max_px - min_px
    width_pixels = max_py - min_py
    
    # Create bounding box corners in PCA space
    corners_pca = np.array([
        [max_px, max_py],
        [max_px, min_py],
        [min_px, max_py], 
        [min_px, min_py]
    ])
    
    # Transform corners back to image space
    corners_img = pca.inverse_transform(corners_pca)
    corners_img = corners_img.astype(int)
    
    # Draw bounding box
    cv.polylines(image, [corners_img], True, (0, 255, 0), 2)
    
    # Calculate center
    center_pca = np.array([(min_px + max_px)/2, (min_py + max_py)/2])
    center_img = pca.inverse_transform(center_pca.reshape(1, -1))[0]
    center = tuple(center_img.astype(int))
    
    # Draw principal axes
    axis_scale = 50
    length_end = pca.inverse_transform([[center_pca[0] + axis_scale, center_pca[1]]])[0]
    width_end = pca.inverse_transform([[center_pca[0], center_pca[1] + axis_scale]])[0]
    
    cv.arrowedLine(image, center, tuple(length_end.astype(int)), (0, 0, 255), 3)  # Red - length
    cv.arrowedLine(image, center, tuple(width_end.astype(int)), (255, 0, 0), 3)   # Blue - width
    
    # Add measurement text
    length_mm = length_pixels * mm_per_pixel
    width_mm = width_pixels * mm_per_pixel
    
    cv.putText(image, f"L:{length_mm:.1f}mm", (center[0]+10, center[1]-20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.putText(image, f"W:{width_mm:.1f}mm", (center[0]+10, center[1]+20),
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

    mm_per_pixel, coin_contour = get_mm_per_pixel(img, contours, ref_size_mm)
    print(f"Millimeters per pixel: {mm_per_pixel}")

    display_img = img.copy()
    results = []

    print(f"Total contours: {len(contours)}")
    filtered = filter_contours_by_area(contours)
    print(f"After area filter: {len(filtered)}")

    for contour in filtered:
        if np.array_equal(contour, coin_contour):
            continue

        length_mm, width_mm, eigenvectors, mean = measure_with_pca(contour, mm_per_pixel)
        draw_pca_measurements(display_img, contour, mm_per_pixel)
        results.append({'length_mm': length_mm, 'width_mm': width_mm})

    plt.figure(figsize=(12, 8))
    plt.imshow(cv.cvtColor(display_img, cv.COLOR_BGR2RGB))
    plt.title("Measured Objects")
    plt.axis("off")
    plt.show()
    

if __name__ == "__main__":
    measure_main("example.jpeg")