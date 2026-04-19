# Aafi Mansuri, Terry Zhen
# Preprocessing pipeline for OMR
# Converts a raw sheet music image into a clean binary image

import cv2
import numpy as np


def binarize(image_path):
    """
    Load a sheet music image, convert to grayscale, and binarize.
    Returns the binary image where black pixels (0) are notation/staff lines
    and white pixels (255) are background.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding automatically determines the optimal threshold
    # THRESH_BINARY_INV so that notation is white (255) and background is black (0)
    # then we invert so notation is black (0) and background is white (255)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    binary = binarize(sys.argv[1])

    cv2.imshow("Binary", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()