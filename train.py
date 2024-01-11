from Augmentation import augment_images
from Distribution import Distribution
import sys
import os

def dataset_augmentation(augmentations_needed):
    for augmentation in augmentations_needed:
        for i in range(augmentation[2], augmentation[1] + augmentation[2]):
            augment_images(augmentation[0] + '/' + str(i) + '.JPG')

def main():
    if len(sys.argv) != 2:
        sys.exit('usage: python3 train.py <directory_path>')

    distribution = Distribution(sys.argv[1])
    augmentations_needed = distribution.augmentations_needed
    for augmentation in augmentations_needed:
        image_in_dir = len(distribution.arborescence[augmentation[0].split('_')[0]][augmentation[0]])
        augmentations_needed[augmentations_needed.index(augmentation)] = (sys.argv[1] + augmentation[0], augmentation[1], image_in_dir)        

    dataset_augmentation(augmentations_needed)

if __name__ == '__main__':
    main()