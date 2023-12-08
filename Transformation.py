from Distribution import Distribution
from PIL import Image, ImageEnhance
import sys
import os
from plantcv import plantcv as pcv
import cv2
import numpy as np

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        value = 128 + factor * (c - 128)
        return max(0, min(255, value))
    return img.point(contrast)

def gaussian_blur(path):    
    image = cv2.imread(path, 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), sigmaX=0)
    pcv.plot_image(gaussian_blur)


def canny_edges(path):
    image = cv2.imread(path, 0)
    gaussian_img = pcv.gaussian_blur(img=image, ksize=(5, 5), sigma_x=1, sigma_y=1)
    edges = cv2.Canny(gaussian_img, threshold1=15, threshold2=90, L2gradient=False)
    Hori = np.concatenate((image, gaussian_img, edges), axis=1)
    
    cv2.imshow('HORIZONTAL', Hori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return ""

def get_mask(path):
    img = cv2.imread(path, 0)
    device = 50
    lp_img = pcv.laplace_filter(img, 1, device)
    # mask, masked_img = pcv.threshold.custom_range(img=img, lower_thresh=[100], upper_thresh=[160], channel='gray')
    pcv.plot_image(lp_img)

def image_transformation(path):
    # leaf_mask(path)
    # gaussian_blur(path)
    canny_edges(path)
    get_mask(path)

def check_path(path):
    if os.path.isdir(path):
        return 0
    try:
        Image.open(path)
        return 1
    except:
        return 2

def main():
    path_type = check_path(sys.argv[1])
    if path_type == 1:
        image_transformation(sys.argv[1])

if __name__ == "__main__":
    main()

