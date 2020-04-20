import numpy as np
import cv2

img = cv2.imread('blobs.jpg',cv2.IMREAD_GRAYSCALE)

detector = cv2.SimpleBlobDetector_create()

keyPoints = detector.detect(img)

print("Blob found: ",len(keyPoints))

cv2.waitKey(0)

