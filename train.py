from Augmentation import augment_images
from Distribution import Distribution
from Transformation import create_transformations
import sys
import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(train_data_dir, val_data_dir, batch_size, epochs, model_save_path, num_classes=8):
    # Image data generators for data augmentation
    train_datagen = ImageDataGenerator(rescale=1./255)
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
    )

    # Flow validation images in batches using val_datagen generator
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
    )

    # num_classes = len(train_generator.class_indices)

    model = create_model((150, 150, 3), num_classes)

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )

    # Save the trained model
    model.save(model_save_path)


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
    if not os.path.exists('./train_augmented/'):
        os.mkdir('./train_augmented/')

    if sys.argv[1][-1] != '/':
        sys.argv[1] += '/'

    distribution = Distribution(sys.argv[1])
    augmentations_needed = distribution.augmentations_needed
    new_subdirectories = []
    for augmentation in augmentations_needed:
        new_subdirectories.append(augmentation[0])
        directory_dest = "./train_augmented/" + augmentation[0] + "/"
        if os.path.exists(directory_dest):
            os.system(f'rm -rf {directory_dest}')
        if not os.path.exists(directory_dest):
            os.mkdir(directory_dest)
            for image in os.listdir(os.path.join(sys.argv[1],augmentation[0])):
                os.system(f'cp \"{sys.argv[1] + augmentation[0]}/{image}\" {directory_dest}')
        image_in_dir = len(distribution.arborescence[augmentation[0].split('_')[0]][augmentation[0]])
        augmentations_needed[augmentations_needed.index(augmentation)] = (sys.argv[1] + augmentation[0], augmentation[1] // 6, image_in_dir, directory_dest)
    for subdirectory in distribution.subdirectories:
        if subdirectory not in new_subdirectories:
            os.system(f'cp -r \"{sys.argv[1] + subdirectory}\" ./train_augmented/')
    dataset_augmentation(augmentations_needed)


if __name__ == '__main__':
    train_data_directory = '../augmented_dataset/'  # Update with your augmented dataset directory
    val_data_directory = '../images/'  # Update with your validation dataset directory
    batch_size = 320
    epochs = 10
    model_save_path = 'leaf_disease_model.h5'
    if len(sys.argv) < 2:
        sys.exit('usage: python3 train.py <directory_path> <flag> -t for training, -a for augmentation')
    if sys.argv[1] == '--train':
        train_model(train_data_directory, val_data_directory, batch_size, epochs, model_save_path)
    elif sys.argv[1] == '--zip':
        shutil.make_archive('dataset', 'zip', "./train_augmented/")
    elif sys.argv[2] == '--augment':
        main()
    elif sys.argv[2] == '--transform':
        create_transformations(["", sys.argv[1], "./train_augmented/" + sys.argv[1].split('/')[-1]])
        # shutil.make_archive('dataset', 'zip', "./train_augmented/")
