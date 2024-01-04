import matplotlib.pyplot as plt
from Distribution import Distribution
from PIL import Image, ImageEnhance
import sys
import os
from plantcv import plantcv as pcv
import cv2
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu, try_all_threshold
from skimage.morphology import closing, square
from skimage.measure import label
from copy import copy

class Transformation:

    def __init__(self, path_to_file):
        self.path_type = self.check_path(path_to_file)
        self.original = self.open_original(path_to_file)
        self.white_balanced_img = pcv.white_balance(self.original, 'max')
        self.contours_img = self.original.copy()
        self.augmented_img = self.augment_image()

        self.mask = 0
        self.binary_mask = 0

        self.canny_edges_contours = 0
        self.canny_edges_img = copy(self.original)


        # gray_img = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        # _, mask = cv2.threshold(gray_img, 85, 255, cv2.THRESH_BINARY)
        # pcv.plot_image(mask)

        self.grey_scale = pcv.rgb2gray(self.original)
        self.augmented_grey_scale = self.convert_augmented_grey_scale()
        self.gaussian_blur_img = 0

        self.corrected_img = 0

        # ret, thresh = cv2.threshold(self.augmented_grey_scale, 127, 255, 0)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)[-2:]
        # cv2.drawContours(self.contours_img, contours, -1, (0,255,0), 3)
        # pcv.plot_image(self.contours_img)
        
        self.mask = 0
        self.b = 0
        self.roi = 0

        if self.path_type == 1:
            self.image_transformation()

        cv2.imshow("canny edges img", self.canny_edges_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # mask = cv2.bitwise_not(self.canny_edges_img)
        # result = cv2.bitwise_and(self.original, self.original, mask=mask)
        # pcv.plot_image(result)

        # pcv.plot_image(pcv.apply_mask(img=self.original, mask=self.canny_edges_img, mask_color='white'))

        # pcv.plot_image(self.original)
        # pcv.plot_image(self.white_balanced_img)
        # pcv.plot_image(self.augmented_img)
        # pcv.plot_image(self.gaussian_blur_img)
        # pcv.plot_image(self.mask)
        # pcv.plot_image(self.b)

    def extract_leaf_from_background(self):
        img = self.original
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # find the green color 
        mask_green = cv2.inRange(hsv, (36,0,0), (86,255,255))
        # find the brown color
        mask_brown = cv2.inRange(hsv, (11, 60, 50), (30, 255, 255))
        # find the yellow color in the leaf
        mask_yellow = cv2.inRange(hsv, (21, 39, 10), (40, 255, 255))
        # find the orange color
        mask_red = cv2.inRange(hsv, (1, 60, 0), (25, 255, 255))

        # find any of the three colors(green or brown or yellow) in the image
        mask = cv2.bitwise_or(mask_green, mask_brown)
        mask = cv2.bitwise_or(mask, mask_yellow)
        mask = cv2.bitwise_or(mask, mask_red)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img,img, mask= mask)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                if np.any(res[i, j] == [0, 0, 0]):
                    res[i, j] = [255, 255 ,255]
        self.mask = res

        cv2.imshow("mask green", mask_green)
        cv2.imshow("mask red", mask_red)
        cv2.imshow("mask yelloz", mask_yellow)
        cv2.imshow("mask brown", mask_brown)
        cv2.imshow("original", img)
        cv2.imshow("final image", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def convert_augmented_grey_scale(self):
        return cv2.cvtColor(self.augmented_img, cv2.COLOR_BGR2GRAY)


    def open_greyscale(self, path_to_file):
        return cv2.imread(path_to_file, 0)


    def open_original(self, path_to_file):
        return cv2.imread(path_to_file)
    

    def color_correction(self):
        threshold_light = pcv.threshold.binary(gray_img=self.gaussian_blur_img, threshold=160, object_type='light')
        # pcv.plot_image(threshold_light)


    def image_transformation(self):
        self.augment_image()
        self.gaussian_blur()
        self.canny_edges()
        self.color_correction()
        self.get_mask()
        # self.get_img_mask()
        self.extract_leaf_from_background()
        self.create_binary_mask()
        self.get_roi()
        self.analyze_size()
        pcv.params.debug = "plot"
        hist_figure, hist_data = pcv.visualize.histogram(img=self.original, hist_data=True, title="Color Histogram")
        hist_df = pd.DataFrame(hist_data)
        hist_df.plot(kind='bar')


    def analyze_size(self):
        labeled_mask, num_seeds = pcv.create_labels(self.binary_mask)
        shape_image = pcv.analyze.size(self.original, labeled_mask=labeled_mask)
        cv2.imshow("analyze size", shape_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_binary_mask(self):
        binary_mask = copy(self.mask)
        for i in range(binary_mask.shape[0]):
            for j in range(binary_mask.shape[1]):
                if np.any(binary_mask[i, j] != [255,255, 255]):
                    binary_mask[i, j] = [0, 0 ,0]
                else:
                    binary_mask[i, j] = [255, 255 ,255]
        self.binary_mask = binary_mask
        cv2.imshow("binary mask", self.binary_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def augment_image(self):
        img = self.white_balanced_img
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
        if len(self.binary_mask.shape) == 3:
            self.binary_mask = cv2.cvtColor(self.binary_mask, cv2.COLOR_BGR2GRAY)
        if self.original.shape[:2] != self.binary_mask.shape:
            self.binary_mask = cv2.resize(self.binary_mask, (self.original.shape[1], self.original.shape[0]))

        roi = cv2.bitwise_and(self.original, self.original, mask=self.binary_mask)
        self.roi = roi
        cv2.imshow("roi", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def gaussian_blur(self):
        grey_img = self.augmented_grey_scale
        gaussian_blur = cv2.GaussianBlur(grey_img, (11, 11), sigmaX=0)
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

    def canny_edges(self):
        self.canny_edges_contours = pcv.canny_edge_detect(self.gaussian_blur_img)
        contours, _ = cv2.findContours(self.canny_edges_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.canny_edges_img, contours, -1, (0, 255, 0), 2)
        # pcv.plot_image(self.original)


    def get_mask(self):
       ()


def main():
    image = Transformation(sys.argv[1])

if __name__ == "__main__":
    main()

