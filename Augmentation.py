from PIL import Image, ImageFilter
import numpy as np
from skimage import transform
from skimage.util import img_as_ubyte
import os
import sys


def flip_image(image):
    return Image.fromarray(np.fliplr(image))


def rotate_image(image):
   rotated_image = transform.rotate(image, 45) # Rotate by 45 degrees
   return Image.fromarray(img_as_ubyte(rotated_image))


def skew_image(image):
   skew_type = transform.AffineTransform(shear=np.pi/6) # Skew with a 30 degree angle
   skewed_image = transform.warp(image, skew_type)
   return Image.fromarray(img_as_ubyte(skewed_image))


def shear_image(image):
   shear_type = transform.AffineTransform(shear=np.pi/4) # Shear with a 45 degree angle
   sheared_image = transform.warp(image, shear_type)
   return Image.fromarray(img_as_ubyte(sheared_image))


def crop_image(image):
    return image.crop((image.width//4, image.height//4, 3*image.width//4, 3*image.height//4))  # Crop the center of the image


def distort_image(image):
    return image.filter(ImageFilter.BLUR)  # Example of a simple distortion


def main():
	if len(sys.argv) != 2:
		sys.exit('wrong number of arguments')

	image_path = sys.argv[1]
	image_dir, image_name = os.path.split(image_path)
	image_name_no_ext = os.path.splitext(image_name)[0]

	image = Image.open(image_path)
	image_np = np.array(image)

	operations = [
		(flip_image, "Flip"),
		(rotate_image, "Rotate"),
		(skew_image, "Skew"),
		(shear_image, "Shear"),
		(crop_image, "Crop"),
		(distort_image, "Distortion"),
	]

	augmented_dir = "augmented/"

	for operation, suffix in operations:
		output_image = operation(image_np if operation in [rotate_image, skew_image, shear_image] else image)
		output_path = os.path.join(augmented_dir, f"{image_name_no_ext}_{suffix}.JPG")
		output_image.save(output_path)


if __name__ == '__main__':
	main()