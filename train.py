from Augmentation import augment_images
from Distribution import Distribution
import sys
import os
import random
def dataset_augmentation(augmentations_needed):
   # augmentations_needed: array of tuples: (images_path, number of images needed, number of images in the directory, directory_dest)

    image_created = 0
    for augmentation in augmentations_needed:
        # create directory if not exists
        index_range = [i for i in range(augmentation[2])]
        for _ in range(augmentation[1]):
            if not index_range:
                break
            index = random.choice(index_range)
            image_path = os.listdir(augmentation[0])[index - 1] # subtract 1 because list indices start at 0
            augment_images(augmentation[0] + '/' + image_path, augmentation[3])
            index_range.remove(index)
        print(f'{augmentation[0]}: {augmentation[1] * 6} images added')
        image_created += augmentation[1]
    print(f'{image_created * 6} images created')


def main():
    if len(sys.argv) != 2:
        sys.exit('usage: python3 train.py <directory_path>')

    if not os.path.exists('./train_augmented/'):
        os.mkdir('./train_augmented/')

    distribution = Distribution(sys.argv[1])
    augmentations_needed = distribution.augmentations_needed
    for augmentation in augmentations_needed:
        directory_dest = "./train_augmented/" + augmentation[0] + "/"
        if not os.path.exists(directory_dest):
            os.mkdir(directory_dest)
        image_in_dir = len(distribution.arborescence[augmentation[0].split('_')[0]][augmentation[0]])
        augmentations_needed[augmentations_needed.index(augmentation)] = (sys.argv[1] + augmentation[0], augmentation[1] // 6, image_in_dir, directory_dest)        
    # print(augmentations_needed)
    dataset_augmentation(augmentations_needed)

if __name__ == '__main__':
    main()