import cv2
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


def register_images(img1_path, img2_path):
    # 1. Load images in Grayscale
    # Using 0 flag loads directly as grayscale
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    # if img1 is None or img2 is None:
    #     print("Error loading images.")
    #     return

    # 2. Preprocessing (Optional but recommended for fluorescence)
    # CLAHE normalizes contrast, helping features pop out despite illumination diffs
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img1_enhanced = clahe.apply(img1)
    img2_enhanced = clahe.apply(img2)

    # 3. Detect Features (SIFT is robust to scale/rotation)
    # Note: If SIFT is slow, you can try ORB: sift = cv2.ORB_create()
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1_enhanced, None)
    kp2, des2 = sift.detectAndCompute(img2_enhanced, None)

    # 4. Match Features using FLANN matcher (faster than Brute Force for large sets)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 5. Filter Matches (Lowe's Ratio Test)
    # Keep only matches where the first match is significantly better than the second
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # if len(good_matches) < 4:
    #     print("Not enough matches found to compute homography.")
    #     return

    # 6. Compute Homography with RANSAC
    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find the transformation matrix (Homography)
    # RANSAC will ignore the outliers (cells that don't match)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 7. Warp Image 1 to align with Image 2
    h, w = img2.shape
    img1_aligned = cv2.warpPerspective(img1, M, (w, h))

    # 8. Visualization (Overlay)
    # Create a color composite: Image 1 (Aligned) in Green, Image 2 in Red
    # This highlights overlapping areas in Yellow
    merged = np.zeros((h, w, 3), dtype=np.uint8)
    merged[:, :, 1] = img1_aligned  # Green Channel
    merged[:, :, 2] = img2  # Red Channel

    # Display results
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:20], None, flags=2))
    plt.title(f"Top 20 Matches (Total Good: {len(good_matches)})")

    plt.subplot(1, 2, 2)
    plt.imshow(merged)
    plt.title("Overlay: Green=Img1(Warped), Red=Img2, Yellow=Overlap")
    plt.show()

    return M, img1_aligned


register_images(r"C:\Users\lzhu\Desktop\oct.png", r"C:\Users\lzhu\Desktop\nc.png")