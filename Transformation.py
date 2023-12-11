from Distribution import Distribution
from PIL import Image, ImageEnhance
import sys
import os
from plantcv import plantcv as pcv
import cv2
import numpy as np


class Transformation:

    def __init__(self, path_to_file):
        self.path_type = self.check_path(path_to_file)
        self.original = self.open_original(path_to_file)
        pcv.plot_image(self.original)
        if self.path_type == 1:
            self.image_transformation()

    def image_transformation(self):
        self.get_mask(self.original)

    def check_path(self, path_to_file):
        if os.path.isdir(path_to_file):
            return 0
        try:
            Image.open(path_to_file)
            return 1
        except:
            return 2

    def open_original(self, path_to_file):
        return cv2.imread(path_to_file)

    def gaussian_blur(self, path):    
        image = cv2.imread(path, 0)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), sigmaX=0)
        pcv.plot_image(gaussian_blur)

    def canny_edges(self, path):
        image = cv2.imread(path, 0)
        gaussian_img = pcv.gaussian_blur(img=image, ksize=(5, 5), sigma_x=1, sigma_y=1)
        edges = cv2.Canny(gaussian_img, threshold1=15, threshold2=90, L2gradient=False)
        Hori = np.concatenate((image, gaussian_img, edges), axis=1)
        
        cv2.imshow('HORIZONTAL', Hori)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return ""

    def get_mask(self, image):
        pass

def main():
    image = Transformation(sys.argv[1])

if __name__ == "__main__":
    main()

