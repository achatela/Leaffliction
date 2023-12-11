from Distribution import Distribution
from PIL import Image, ImageEnhance
import sys
import os
from plantcv import plantcv as pcv
import cv2
import numpy as np
from skimage.filters import threshold_otsu, try_all_threshold
from skimage.morphology import closing
from skimage.measure import label


class Transformation:

    def __init__(self, path_to_file):
        self.path_type = self.check_path(path_to_file)
        self.original = self.open_original(path_to_file)
        self.grey_scale = self.open_greyscale(path_to_file)
        self.gaussian_blur_img = 0
        self.mask = 0
        self.b = 0

        if self.path_type == 1:
            self.image_transformation()
        pcv.plot_image(self.original)
        pcv.plot_image(self.mask)
        pcv.plot_image(self.b)

    def open_greyscale(self, path_to_file):
        return cv2.imread(path_to_file, 0)


    def open_original(self, path_to_file):
        return cv2.imread(path_to_file)
    

    def image_transformation(self):
        self.get_mask(self.original)
        self.gaussian_blur()
        self.get_img_mask()


    def check_path(self, path_to_file):
        if os.path.isdir(path_to_file):
            return 0
        try:
            Image.open(path_to_file)
            return 1
        except:
            return 2


    def gaussian_blur(self):
        grey_img = self.grey_scale
        gaussian_blur = cv2.GaussianBlur(grey_img, (5, 5), sigmaX=0)
        self.gaussian_blur_img = gaussian_blur

    def get_img_mask(self):
        img = self.grey_scale
        # try_all_threshold(img)
        global_thresh = threshold_otsu(img)
        binary_global = img < global_thresh
        binary_global = closing(binary_global)
        label_img = label(binary_global, background='black')
        self.mask = label_img
        self.b = binary_global

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

