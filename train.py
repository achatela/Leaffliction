from Augmentation import augment_images
from Distribution import Distribution
import sys
import os

def main():
    if len(sys.argv) != 2:
        sys.exit('usage: python3 train.py <directory_path>')

    distribution = Distribution(sys.argv[1])
    augmentations_needed = distribution.augmentations_needed
    for augmentation in augmentations_needed:
        augmentations_needed[augmentations_needed.index(augmentation)] = (sys.argv[1] + augmentation[0], augmentation[1])        

    for augmentation in augmentations_needed:
        augment_images(augmentation[0], augmentation[1])

if __name__ == '__main__':
    main()