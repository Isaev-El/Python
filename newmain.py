# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Исправлена ошибка в настройке уровня логирования
#
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.datasets import mnist
# from tensorflow import keras
# from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train / 255
# x_test = x_test / 255
#
# y_train_cat = keras.utils.to_categorical(y_train, 10)
# y_test_cat = keras.utils.to_categorical(y_test, 10)
#
# x_train = np.expand_dims(x_train, axis=3)
# x_test = np.expand_dims(x_test, axis=3)
#
# print(x_train.shape)
#
# model = keras.Sequential([
#     Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),  # Исправлена ошибка
#     MaxPooling2D((2, 2), strides=2),
#     Conv2D(32, (3, 3), padding='same', activation='relu'),
#     MaxPooling2D((2, 2), strides=2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax'),
# ])
#
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# history = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
#
# model.evaluate(x_test, y_test_cat)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.datasets import mnist
import cv2
from tensorflow import keras
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)


model = keras.Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(x_train, y_train_cat, batch_size=32, epochs=10, validation_split=0.2)

# Загружаем фотки для предсказания
image_path = 'chislo.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Делаем предсказание
predictions = model.predict(image)
predicted_class = np.argmax(predictions)
confidence = predictions[0][predicted_class]

# Выводим результат
print("Predicted Class:", predicted_class)
print("Confidence:", confidence)

model.evaluate(x_test, y_test_cat)
