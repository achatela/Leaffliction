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
    # image = Image.fromarray(image)
    # enhancer = ImageEnhance.Contrast(image)
    # img = enhancer.enhance(1.5)
    # img = change_contrast(img, 100)
    # img = np.array(img)

    sobel = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=13)
    # edges = cv2.Canny(image, threshold1=1, threshold2=120, L2gradient=False)
    Hori = np.concatenate((image, sobel), axis=1) 
    
    cv2.imshow('HORIZONTAL', Hori) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return ""

def image_transformation(path):
    # img_data= cv2.imread(path)
    # pcv.plot_image(img_data)
    gaussian_blur(path)
    canny_edges(path)

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

