from PIL import Image
import sys
import os


def main():
    if len(sys.argv) != 2:
        sys.exit('wrong number of arguments')

    highest = float("-inf")
    subdirectories = set()
    categories = set()
    images_path = []

    for path, subdirs, files in os.walk(sys.argv[1]):
        subdirectories.update(subdirs)
        for name in files:
            images_path.append(os.path.join(path, name))
    for subdirectory in subdirectories:
        categories.update([subdirectory.split('_')[0]])

    image_path = images_path[0]
    image_name = image_path.split('/')[-1]
    Original_Image = Image.open(image_path)

    # Rotate Image By 180 Degree 
    rotated_image = Original_Image.rotate(45)

    rotated_image.save(f'augmented/{image_name[: -3]}_Rotate.JPG')

    rotated_image.show()


if __name__ == '__main__':
    main()