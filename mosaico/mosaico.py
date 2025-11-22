import cv2
import numpy as np

def sift_match(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise ValueError("Error: No SIFT features detected in one of the images")

    # FLANN parameters for SIFT
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    print(f"Good matches = {len(good)}")

    if len(good) < 10:
        raise ValueError(f"ERROR: Not enough matches! Only {len(good)} good matches.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Compute homography from img2 to img1
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError("Could not compute homography!")

    # Count inliers
    matches_mask = mask.ravel().tolist()
    inliers = matches_mask.count(1)
    print(f"Inliers = {inliers}/{len(good)}")

    return H

def warp(img_ref, img_to_warp, H):
    h1, w1 = img_ref.shape[:2]
    h2, w2 = img_to_warp.shape[:2]

    # Corners of the image to warp
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners2, H)

    # Combine corners
    all_corners = np.concatenate([
        warped_corners,
        np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    ])

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    # Translation matrix to avoid negative coordinates
    tx, ty = -xmin, -ymin
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # Warp the second image
    result = cv2.warpPerspective(img_to_warp, T @ H, (xmax - xmin, ymax - ymin))

    # Blend the reference image (simple overlay for now)
    result[ty:ty + h1, tx:tx + w1] = img_ref

    return result

def stitch_around_center(images):
    if len(images) < 2:
        raise ValueError("Need at least 2 images to stitch")

    center_idx = len(images) // 2
    panorama = images[center_idx].copy()

    print(f"Using image {center_idx + 1} as center reference")

    # Stitch images to the right of center
    for i in range(center_idx + 1, len(images)):
        print(f"Stitching image {i + 1} to the right...")
        try:
            H = sift_match(panorama, images[i])
            panorama = warp(panorama, images[i], H)
            print(f"Successfully stitched image {i + 1}")
        except Exception as e:
            print(f"Failed to stitch image {i + 1}: {e}")
            continue

    # Stitch images to the left of center
    for i in range(center_idx - 1, -1, -1):
        print(f"Stitching image {i + 1} to the left...")
        try:
            H = sift_match(panorama, images[i])
            panorama = warp(panorama, images[i], H)
            print(f"Successfully stitched image {i + 1}")
        except Exception as e:
            print(f"Failed to stitch image {i + 1}: {e}")
            continue

    return panorama

def main():
    # Load images
    print("Loading images...")
    img1 = cv2.imread("input/img1.jpeg")
    img2 = cv2.imread("input/img2.jpeg")
    img3 = cv2.imread("input/img3.jpeg")
    img4 = cv2.imread("input/img4.jpeg")

    # Check if images loaded properly
    images = [img1, img2, img3, img4]
    if any(img is None for img in images):
        raise FileNotFoundError("One or more input images not found")

    print(f"Loaded {len(images)} images successfully")

    try:
        print("\n=== Using Center-based Stitching ===")
        panorama = stitch_around_center(images)

        cv2.imwrite("output/panorama_result.jpg", panorama)
        print(f"\nSuccess! Panorama saved: output/panorama_result.jpg")
        print(f"Final panorama size: {panorama.shape[1]} x {panorama.shape[0]}")

    except Exception as e:
        print(f"Error during stitching: {e}")


if __name__ == "__main__":
    main()
