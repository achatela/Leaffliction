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
from rembg import remove

class Transformation:

    def __init__(self, path_to_file):
        self.path_type = self.check_path(path_to_file)
        self.original = self.open_original(path_to_file)
        self.white_balanced_img = pcv.white_balance(self.original, 'max')
        self.contours_img = self.original.copy()
        self.augmented_img = self.augment_image()
        self.pseudolandmarks_img = 0

        self.binary_mask = 0

        self.canny_edges_contours = 0
        self.canny_edges_img = copy(self.original)

        self.grey_scale = pcv.rgb2gray(self.original)
        self.augmented_grey_scale = self.convert_augmented_grey_scale()
        self.gaussian_blur_img = 0

        self.corrected_img = 0

        self.mask = 0
        self.b = 0
        self.roi = 0
        self.shape_image = 0
        if self.path_type == 1:
            self.image_transformation()
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2RGB)
        self.pseudolandmarks_img = cv2.cvtColor(self.pseudolandmarks_img, cv2.COLOR_BGR2RGB)
        self.shape_image = cv2.cvtColor(self.shape_image, cv2.COLOR_BGR2RGB)
        


    def extract_leaf_from_background(self):
        img = copy(self.original)
        res = remove(img)
        self.mask = res

    def convert_augmented_grey_scale(self):
        return cv2.cvtColor(self.augmented_img, cv2.COLOR_BGR2GRAY)


    def open_greyscale(self, path_to_file):
        return cv2.imread(path_to_file, 0)


    def open_original(self, path_to_file):
        return cv2.imread(path_to_file)

    def image_transformation(self):
        self.augment_image()
        self.gaussian_blur()
        self.canny_edges()
        self.extract_leaf_from_background()
        self.create_binary_mask()
        self.get_roi()
        self.analyze_size()
        self.pseudolandmarks()

        hist_figure, hist_data = pcv.visualize.histogram(img=self.original, hist_data=True, title="Color Histogram")
        # plt.show()
        # hist_df = pd.DataFrame(hist_data)
        # hist_df.plot(kind='bar')


    def pseudolandmarks(self):
        img = copy(self.original)
        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=img, mask=self.binary_mask, label="default")

        # Draw points on the image
        radius = 5  # Adjust as needed
        dark_blue = [139,0, 0]  # Dark blue color
        orange = (0, 165, 255)  # Orange color
        pink = (180, 105, 255) # Pink color
        thickness = -1  # To fill the circle

        for circle in top:
            cv2.circle(img, [int(circle[0][0]), int(circle[0][1])], radius, dark_blue, thickness)
        for circle in bottom:
            cv2.circle(img, [int(circle[0][0]), int(circle[0][1])], radius, pink, thickness)
        for circle in center_v:
            cv2.circle(img, [int(circle[0][0]), int(circle[0][1])], radius, orange, thickness)

        self.pseudolandmarks_img = img

    def analyze_size(self):
        labeled_mask, num_seeds = pcv.create_labels(self.binary_mask)
        self.shape_image = pcv.analyze.size(self.original, labeled_mask=labeled_mask)

    def create_binary_mask(self):
        binary_mask = copy(self.mask)
        for i in range(binary_mask.shape[0]):
            for j in range(binary_mask.shape[1]):
                if np.any(binary_mask[i, j] != [0,0,0,0]):
                    binary_mask[i, j] = [255,255,255,255]
                else:
                    binary_mask[i, j] = [0,0,0,0]
        self.binary_mask = binary_mask

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

    def gaussian_blur(self):
        grey_img = self.augmented_grey_scale
        gaussian_blur = cv2.GaussianBlur(grey_img, (11, 11), sigmaX=0)
        self.gaussian_blur_img = gaussian_blur

    def canny_edges(self):
        self.canny_edges_contours = pcv.canny_edge_detect(self.gaussian_blur_img)
        contours, _ = cv2.findContours(self.canny_edges_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.canny_edges_img, contours, -1, (0, 255, 0), 2)


def create_transformations(arg1, arg2):
    if '.JPG' in arg1:
        transformation = Transformation(arg1)
        cv2.imshow("Original", transformation.original)
        cv2.imshow("Gaussan Blur", transformation.gaussian_blur_img)
        cv2.imshow("Mask", transformation.mask)
        cv2.imshow("Binary Mask", transformation.binary_mask)
        cv2.imshow("Analyze Object", transformation.shape_image)
        cv2.imshow("Pseudolandmarks", transformation.pseudolandmarks_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if len(sys.argv) < 3:
            print("usage: python3 Transformation.py <input_dir> <dest_dir>")
            return
        if not os.path.isdir(arg2):
            print("dest_dir is not a dir")
            return

        filenames = []
        for path, subdirs, files in os.walk(arg1):
            for file in files:
                if '.JPG' in file:
                    file_path = os.path.join(path, file)
                    file_path = file_path.replace('.JPG', '')
                    transformation = Transformation(file_path + '.JPG')
                    filename = file.replace('.JPG', '')
                    Image.fromarray(transformation.gaussian_blur_img).save(arg2 + filename + '-gaussian.JPG')
                    Image.fromarray(transformation.mask).save(arg2 + filename + '-mask.JPG')
                    Image.fromarray(transformation.binary_mask).convert('RGB').save(arg2 + filename + '-binary-mask.JPG')
                    Image.fromarray(transformation.shape_image).convert('RGB').save(arg2 + filename + '-analyze-object.JPG')
                    Image.fromarray(transformation.pseudolandmarks_img).convert('RGB').save(arg2 + filename + '-pseudolandmarks.JPG')

def main():
    if len(sys.argv) < 2:
        print("usage: python3 Transformation.py <input_img> or python3 Transformation.py <input_dir> <dest_dir>")
        return
    if len(sys.argv) == 2:
        create_transformations(sys.argv[1], "")
    else:
        create_transformations(sys.argv[1], sys.argv[2])
        
if __name__ == "__main__":
    main()

