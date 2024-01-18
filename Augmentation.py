import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
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
    if random.randint(0, 1):
        degrees = random.randint(10, 90)
    else:
        degrees = random.randint(-90, -10)
    return image.rotate(degrees, expand=True)


def skew_image(image):
    if random.randint(0, 1):
        skew_factor_x = random.randint(1, 5) / 10
    else:
        skew_factor_x = random.randint(-5, -1) / 10
    skew_factor_y = 0

    matrix = (1, skew_factor_x, 0, skew_factor_y, 1, 0)

    return image.transform(image.size,
                           Image.AFFINE, matrix, resample=Image.BICUBIC)


def shear_image(image):
    shear_factor_x = 0
    if random.randint(0, 1):
        shear_factor_y = random.randint(1, 5) / 10
    else:
        shear_factor_y = random.randint(-5, -1) / 10

    return image.transform(
        image.size,
        Image.AFFINE,
        (1, shear_factor_x, 0, shear_factor_y, 1, 0),
        resample=Image.BICUBIC
    )


def crop_image(image):
    width, height = image.size
    crop_width = random.randint(round(width*0.5), round(width*0.8))
    crop_height = random.randint(round(height*0.5), round(height*0.8))
    left = random.randint(0, width - crop_width)
    upper = random.randint(0, height - crop_height)
    right = left + crop_width
    lower = upper + crop_height
    return image.crop((left, upper, right, lower))


def distort_image(np_image):
    aug = iaa.PerspectiveTransform(scale=(0.1, 0.16))
    return aug.augment_image(np_image)


def augment_images(image_path, path_dest):
    image_dir, image_name = os.path.split(image_path)
    image_name, image_ext = os.path.splitext(image_name)
    print(image_name)

    if not os.path.exists(image_path):
        sys.exit(f'"{image_path}" does not exist')
    if image_ext != '.JPG':
        sys.exit(f'"{image_path}" is not a JPG file')
    try:
        image = Image.open(image_path)
        np_image = np.array(image)
    except Exception:
        sys.exit(f'"{image_path}" is not a JPG file')

    augmentations = [
        (flip_image, "Flip"),
        (rotate_image, "Rotate"),
        (skew_image, "Skew"),
        (shear_image, "Shear"),
        (crop_image, "Crop"),
        (distort_image, "Distortion"),
    ]

    for function, name in augmentations:
        if name == 'Flip' or name == 'Rotate' or name == 'Skew' \
          or name == 'Shear' or name == 'Crop':
            augmented_image = function(image)
            if path_dest == "":
                augmented_image.show()
                augmented_image.save(f'augmented/{image_name}_{name}.JPG')
            else:
                augmented_image.save(f'{path_dest}{image_name}_{name}.JPG')
        else:
            augmented_image = function(np_image)
            if name != 'Distortion':
                augmented_image = \
                    (augmented_image * 255).astype(np.uint8)
            if path_dest == "":
                Image.fromarray(augmented_image) \
                    .save(f'augmented/{image_name}_{name}.JPG')
                plt.imshow(augmented_image)
                plt.show()
            else:
                Image.fromarray(augmented_image) \
                    .save(f'{path_dest}{image_name}_{name}.JPG')


def main():
    if len(sys.argv) != 2:
        sys.exit('wrong number of arguments')
    if not os.path.exists('./augmented'):
        os.system('rm -rf ./augmented')
        os.mkdir('./augmented/')
    else:
        os.system('rm -rf ./augmented')
        os.mkdir('./augmented/')

    augment_images(sys.argv[1], "")


if __name__ == '__main__':
    main()
