import os
import csv
import cv2
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense


BATCH_SIZE = 32

def generator(samples, batch_size=32):
    num_samples = len(samples)
    batch_size = batch_size // 6

    while True:
        samples = shuffle(samples)
        
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images, angles = [], []
            correction = [0.0, 0.2, -0.2]

            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])

                for i in range(3):
                    name = './IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    image = image[50:-20,:,:]
                    image = cv2.resize(image, (200,66))
                    angle = center_angle + correction[i]

                    image_flipped = np.fliplr(image)
                    angle_flipped = -angle

                    images.append(image)
                    angles.append(angle)
                    images.append(image_flipped)
                    angles.append(angle_flipped)

            X_train, y_train = np.array(images), np.array(angles)
            yield shuffle(X_train, y_train)


if __name__ == "__main__":
    samples = []
    with open('./driving_log.csv') as file:
        reader = csv.reader(file)
        next(reader)
        for line in reader:
            samples.append(line)

    samples = shuffle(samples)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, BATCH_SIZE)
    validation_generator = generator(validation_samples, BATCH_SIZE)

    # Model architecture
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66,200,3)))
    model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(Dropout((0.5)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, steps_per_epoch=len(train_samples), 
                        validation_data=validation_generator, 
                        validation_steps=len(validation_samples), 
                        epochs=10, verbose=1)
    model.save('model.h5')