from PIL import Image
import sys
import os
from plantcv import plantcv as pcv
import cv2
import numpy as np
from copy import copy
from rembg import remove

class Transformation:

    def __init__(self, path_to_file):
        self.path_type = self.check_path(path_to_file)
        self.original = self.open_original(path_to_file)
        self.white_balanced_img = pcv.white_balance(self.original, 'max')
        self.augmented_img = self.augment_image()
        self.contours_img = self.original.copy()
        self.pseudolandmarks_img = self.original.copy()

        self.binary_mask = 0

        self.canny_edges_contours = 0
        self.canny_edges_img = self.original.copy()

        self.grey_scale = pcv.rgb2gray(self.original)
        self.augmented_grey_scale = cv2.cvtColor(self.augmented_img, cv2.COLOR_BGR2GRAY)
        self.gaussian_blur_img = 0

        self.corrected_img = 0

        self.mask = 0
        self.b = 0
        self.roi = 0
        self.shape_image = 0

        if self.path_type == 1:
            self.image_transformation()

        # self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2RGB)
        # self.pseudolandmarks_img = cv2.cvtColor(self.pseudolandmarks_img, cv2.COLOR_BGR2RGB)
        # self.shape_image = cv2.cvtColor(self.shape_image, cv2.COLOR_BGR2RGB)
        

    def augment_image(self):
        # converting to LAB color space
        lab= cv2.cvtColor(self.white_balanced_img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))

        # Converting image from LAB Color model to BGR color spcae
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


    def open_original(self, path_to_file):
        return cv2.imread(path_to_file)
    

    def image_transformation(self):
        self.gaussian_blur()
        self.extract_leaf_from_background()
        self.create_binary_mask()
        self.get_roi()
        self.analyze_size()
        self.pseudolandmarks()
        self.canny_edges()


    def gaussian_blur(self):
        self.gaussian_blur_img = cv2.GaussianBlur(self.augmented_grey_scale, (11, 11), sigmaX=0)


    def extract_leaf_from_background(self):
        self.mask = remove(self.original)


    def create_binary_mask(self):
        self.binary_mask = copy(self.mask)
        for i in range(self.binary_mask.shape[0]):
            for j in range(self.binary_mask.shape[1]):
                if np.any(self.binary_mask[i, j] != [0,0,0,0]):
                    self.binary_mask[i, j] = [255,255,255,255]
                else:
                    self.binary_mask[i, j] = [0,0,0,0]
        

    def get_roi(self):
        if len(self.binary_mask.shape) == 3:
            self.binary_mask = cv2.cvtColor(self.binary_mask, cv2.COLOR_BGR2GRAY)

        if self.original.shape[:2] != self.binary_mask.shape:
            self.binary_mask = cv2.resize(self.binary_mask, (self.original.shape[1], self.original.shape[0]))

        self.roi = cv2.bitwise_and(self.original, self.original, mask=self.binary_mask)


    def analyze_size(self):
        labeled_mask, num_seeds = pcv.create_labels(self.binary_mask)
        self.shape_image = pcv.analyze.size(self.original, labeled_mask=labeled_mask)


    def pseudolandmarks(self):
        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=self.pseudolandmarks_img, mask=self.binary_mask, label="default")

        # Draw points on the image
        radius = 5  # Adjust as needed
        dark_blue = [139,0, 0]  # Dark blue color
        orange = (0, 165, 255)  # Orange color
        pink = (180, 105, 255) # Pink color
        thickness = -1  # To fill the circle

        for i in range(len(top)):
            cv2.circle(self.pseudolandmarks_img, [int(top[i][0][0]), int(top[i][0][1])], radius, dark_blue, thickness)
            cv2.circle(self.pseudolandmarks_img, [int(bottom[i][0][0]), int(bottom[i][0][1])], radius, pink, thickness)
            cv2.circle(self.pseudolandmarks_img, [int(center_v[i][0][0]), int(center_v[i][0][1])], radius, orange, thickness)


    def canny_edges(self):
        self.canny_edges_contours = pcv.canny_edge_detect(self.mask)
        contours, _ = cv2.findContours(self.canny_edges_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.canny_edges_img, contours, -1, (0, 255, 0), 2)


    def check_path(self, path_to_file):
        if os.path.isdir(path_to_file):
            return 0
        try:
            Image.open(path_to_file)
            return 1
        except:
            return 2

def create_transformations(argv):
    if len(argv) == 2:
        file_path = argv[1]
        if not os.path.exists(file_path):
            sys.exit(f'"{file_path}" does not exist')
        _, file_ext = os.path.splitext(file_path)
        if file_ext != '.JPG' or not os.path.isfile(file_path):
            sys.exit(f'"{file_path}" is not a JPG file')
        try:
            transformation = Transformation(file_path)
        except:
            sys.exit(f'"{file_path}" is not a JPG file')
        cv2.imshow("Original", transformation.original)
        cv2.imshow("Gaussan Blur", transformation.gaussian_blur_img)
        cv2.imshow("Mask", transformation.mask)
        cv2.imshow("Binary Mask", transformation.binary_mask)
        cv2.imshow("Analyze Object", transformation.shape_image)
        cv2.imshow("Pseudolandmarks", transformation.pseudolandmarks_img)
        cv2.imshow("Canny Edges", transformation.canny_edges_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif len(argv) == 3:
        src_dir = argv[1]
        if not os.path.exists(src_dir):
            sys.exit(f'"{src_dir}" does not exist')
        if not os.path.isdir(src_dir):
            sys.exit(f'"{src_dir}" is not a directory')
        dst_dir = argv[2]
        if os.path.exists(dst_dir):
            if not os.path.isdir(dst_dir):
                sys.exit(f'"{dst_dir}" is not a directory')
        else:
            os.mkdir(dst_dir)
        for file in os.listdir(argv[1]):
            file_name, file_ext = os.path.splitext(file)
            file_path = os.path.join(src_dir, file)
            if file_ext != '.JPG' or not os.path.isfile(file_path):
                print(f'"{file}" is not a JPG file')
                continue
            try:
                transformation = Transformation(file_path)
            except:
                print(f'"{file}" is not a JPG file')
                continue
            transformed_file_path_prefix = os.path.join(dst_dir, file_name)
            Image.fromarray(transformation.gaussian_blur_img).save(transformed_file_path_prefix + '-gaussian.JPG')
            Image.fromarray(transformation.mask).convert('RGB').save(transformed_file_path_prefix + '-mask.JPG')
            Image.fromarray(transformation.binary_mask).convert('RGB').save(transformed_file_path_prefix + '-binary-mask.JPG')
            Image.fromarray(transformation.shape_image).convert('RGB').save(transformed_file_path_prefix + '-analyze-object.JPG')
            Image.fromarray(transformation.pseudolandmarks_img).convert('RGB').save(transformed_file_path_prefix + '-pseudolandmarks.JPG')
            Image.fromarray(transformation.canny_edges_img).convert('RGB').save(transformed_file_path_prefix + '-canny-edges.JPG')
    else:
        sys.exit('usage: python3 Transformation.py <input_img> or python3 Transformation.py <input_dir> <dest_dir>')
        
if __name__ == '__main__':
    create_transformations(sys.argv)

