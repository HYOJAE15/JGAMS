from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import random
from matplotlib import pyplot as plt


## [Load image]
parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
args = parser.parse_args()

src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
## [Load image]

## [Convert to grayscale(binary)]
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
## [Convert to grayscale(binary)]

## [Apply Histogram Equalization]
dst = cv.equalizeHist(src)
## [Apply Histogram Equalization]

## [Convert to bgrimage]
dst_bgr = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
## [Convert to bgrimage]

# Calculate Image Histogram 
hist1 = cv.calcHist([src],[0],None,[256],[0,256])
print(f"hist1_type: {type(hist1)}, hist1: {hist1}")
# hist2 = cv.calcHist([dst],[0],None,[256],[0,256])
# hist3 = cv.calcHist([dst_bgr],[0],None,[256],[0,256])

# plt.subplot(221),plt.imshow(src, "gray"),plt.title('src')
# plt.subplot(222),plt.imshow(dst, "gray"),plt.title('dst')
# plt.subplot(223),plt.imshow(dst_bgr, "gray"),plt.title('dst_bgr')
# plt.subplot(224),
hist1Flatten = hist1.flatten()
plt.plot(hist1Flatten)
# plt.plot(hist2,color='g')
# plt.plot(hist3,color='b')
# plt.xlim([0,256])
plt.show()

# Resize Factor
fx_f = 0.2
fy_f = 0.2

# Image Resize
src_r = cv.resize(src, dsize=(0, 0), fx=fx_f, fy=fy_f, interpolation=cv.INTER_AREA)
dst_r = cv.resize(dst, dsize=(0, 0), fx=fx_f, fy=fy_f, interpolation=cv.INTER_AREA)
dst_bgr_r = cv.resize(dst_bgr, dsize=(0, 0), fx=fx_f, fy=fy_f, interpolation=cv.INTER_AREA)

## [Display results]
cv.imshow('Source image', src_r)
cv.imshow('Equalized Image', dst_r)
cv.imshow('Equalized_bgr Image', dst_bgr_r)
## [Display results]

## [Wait until user exits the program]
cv.waitKey()
## [Wait until user exits the program]