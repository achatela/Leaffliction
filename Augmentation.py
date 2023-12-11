import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from skimage import transform
from PIL import Image
import numpy as np
import random
import sys
import os


def flip_image(image):
	if random.randint(0, 1):
		return image.transpose(Image.FLIP_LEFT_RIGHT)
	else:
		return image.transpose(Image.FLIP_TOP_BOTTOM)

def rotate_image(image):
	return image.rotate(random.randint(10, 350), expand=True)

def skew_image(np_image):
	tf = transform.AffineTransform(shear=0.5)
	return transform.warp(np_image, tf.inverse)

def shear_image(np_image):
	tf = transform.AffineTransform(shear=0.2)
	return transform.warp(np_image, tf)

def crop_image(image):
	width, height = image.size
	crop_width = random.randint(round(width*0.1), round(width*0.9))
	crop_height = random.randint(round(height*0.1), round(height*0.9))
	left = random.randint(0, width - crop_width)
	upper = random.randint(0, height - crop_height)
	right = left + crop_width
	lower = upper + crop_height
	return image.crop((left, upper, right, lower))

def distort_image(np_image):
	aug = iaa.PerspectiveTransform(scale=(0.1, 0.2))
	return aug.augment_image(np_image)

def main():
	if len(sys.argv) != 2:
		sys.exit('wrong number of arguments')

	image_path = sys.argv[1]
	image_dir, image_name = os.path.split(image_path)
	image_name = os.path.splitext(image_name)[0]

	image = Image.open(image_path)
	np_image = np.array(image)

	augmentations = [
		(flip_image, "Flip"),
		(rotate_image, "Rotate"),
		(skew_image, "Skew"),
		(shear_image, "Shear"),
		(crop_image, "Crop"),
		(distort_image, "Distortion"),
	]

	for function, name in augmentations:
		if name == 'Flip' or name == 'Rotate' or name == 'Crop':
			augmented_image = function(image)
			augmented_image.show()
			augmented_image.save(f'augmented/{image_name}_{name}.JPG')
		else:
			augmented_image = function(np_image)
			plt.imshow(augmented_image)
			plt.show()
			if name != 'Distortion':
				augmented_image = (augmented_image * 255).astype(np.uint8)
			Image.fromarray(augmented_image).save(f'augmented/{image_name}_{name}.JPG')

if __name__ == '__main__':
	main()