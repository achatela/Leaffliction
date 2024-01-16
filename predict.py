import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.utils import to_categorical

class_names = ["Apple_Black_rot", "Apple_healthy", "Apple_rust", "Apple_scab", "Grape_Black_rot", "Grape_Esca", "Grape_healthy", "Grape_spot"]

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

def load_trained_model(weights_path):
   model = create_model((150, 150, 3), 8)
   model.load_weights(weights_path)
   return model

def main():
    if len(sys.argv) < 2:
        sys.exit("Need args")
    model = load_trained_model(sys.argv[1])
    im = Image.open(sys.argv[2])
    im = im.resize((150, 150))  # Resize the image to the expected input shape
    np_img = np.array(im)
    np_img = np_img.reshape((1, 150, 150, 3))  # Add the batch dimension
    prediction = model.predict(np_img)

    max_positions = [i for i, x in enumerate(prediction[0]) if x == np.max(prediction[0])]
    print(class_names[max_positions[0]])
    # class_predicted = to_categorical(prediction)
    print(prediction)  # You may want to print out the prediction or handle it as needed


if __name__ == "__main__":
    main()