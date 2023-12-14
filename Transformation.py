import matplotlib.pyplot as plt
from Distribution import Distribution
from PIL import Image, ImageEnhance
import sys
import os
from plantcv import plantcv as pcv
import cv2
import numpy as np
from skimage.filters import threshold_otsu, try_all_threshold
from skimage.morphology import closing, square
from skimage.measure import label


class Transformation:

    def __init__(self, path_to_file):
        self.path_type = self.check_path(path_to_file)
        self.original = self.open_original(path_to_file)
        self.contours_img = self.original.copy()
        self.augmented_img = self.augment_image()

        self.grey_scale = self.open_greyscale(path_to_file)
        self.augmented_grey_scale = self.convert_augmented_grey_scale()
        self.gaussian_blur_img = 0

        self.white_balanced_img = pcv.white_balance(self.original, 'hist', roi=[5,5,80,80])
        self.corrected_img = 0

        ret, thresh = cv2.threshold(self.augmented_grey_scale, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)[-2:]
        cv2.drawContours(self.contours_img, contours, -1, (0,255,0), 3)
        pcv.plot_image(self.contours_img)
        
        self.mask = 0
        self.b = 0
        self.roi = 0

        if self.path_type == 1:
            self.image_transformation()

        pcv.plot_image(self.original)
        pcv.plot_image(self.white_balanced_img)
        pcv.plot_image(self.corrected_img)
        pcv.plot_image(self.augmented_img)
        pcv.plot_image(self.gaussian_blur_img)
        pcv.plot_image(self.mask)
        pcv.plot_image(self.b)


    def convert_augmented_grey_scale(self):
        return cv2.cvtColor(self.augmented_img, cv2.COLOR_BGR2GRAY)


    def open_greyscale(self, path_to_file):
        return cv2.imread(path_to_file, 0)


    def open_original(self, path_to_file):
        return cv2.imread(path_to_file)
    

    def color_correction(self):
        ()
        # card_mask = pcv.transform.find_color_card(rgb_img=self.white_balanced_img, radius=15)
        # headers, card_matrix = pcv.transform.get_color_matrix(rgb_img=self.white_balanced_img, mask=card_mask)
        # std_color_matrix = pcv.transform.std_color_matrix(pos=3)
        # img_cc = pcv.transform.affine_color_correction(rgb_img=self.white_balanced_img, source_matrix=card_matrix, 
        #                                        target_matrix=std_color_matrix)
        # self.corrected_img = img_cc


    def image_transformation(self):
        self.color_correction()
        self.augment_image()
        self.gaussian_blur()
        self.get_mask()
        # self.get_roi()
        self.get_img_mask()


    def augment_image(self):
        img = self.original
        # converting to LAB color space
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_img


    def check_path(self, path_to_file):
        if os.path.isdir(path_to_file):
            return 0
        try:
            Image.open(path_to_file)
            return 1
        except:
            return 2
        
    def get_roi(self):
        roi = pcv.roi.rectangle(img=self.original, x=10, y=10, h=244, w=244)
        self.roi = roi

    def gaussian_blur(self):
        grey_img = self.augmented_grey_scale
        gaussian_blur = cv2.GaussianBlur(grey_img, (13, 13), sigmaX=0)
        self.gaussian_blur_img = gaussian_blur

    def get_img_mask(self):
        img = self.gaussian_blur_img
        global_thresh = threshold_otsu(img)
        binary_global = img < global_thresh
        binary_global = closing(binary_global)
        label_img = label(binary_global, background='grey', connectivity=2)
        try_all_threshold(label_img)
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


    def get_mask(self):
       ()


def main():
    image = Transformation(sys.argv[1])

if __name__ == "__main__":
    main()

