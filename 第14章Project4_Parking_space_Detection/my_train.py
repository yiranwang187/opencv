# 使用mobileNet网络进行预测
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

img_width, img_height = 48, 48
train_data_dir = "train_data/train"
validation_data_dir = "train_data/test"


# 导入模型
mobile = tf.keras.applications.mobilenet.MobileNet(include_top=False,input_shape=(img_width, img_height,3))
print(mobile.summary())

# 选择第一层到倒数第六层

# x = mobile.layers[-6].output
# output = Dense(units=2, activation='softmax')(x)

inputs = tf.keras.Input(shape=(img_width, img_height,3))
x = mobile.layers[1](inputs)
for layer in mobile.layers[2:-6]:
    x = layer(x)

# x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
x = tf.keras.layers.Flatten()(x)
output = Dense(units=2, activation='softmax')(x)
# 构建一个新的模型
# model = Model(inputs=mobile.input, outputs=output)
model = Model(inputs=inputs, outputs=output)
# print(model.summary())

# 冻结第一层到倒数23层
# 只训练最后的23层
for layer in model.layers[1:-20]:
    layer.trainable = False
print(model.summary())

# 编译模型
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 准备数据
files_train = 0
files_validation = 0
nb_train_samples = files_train
nb_validation_samples = files_validation
batch_size = 32
epochs = 15
num_classes = 2


cwd = os.getcwd()
folder = 'train_data/train'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
    files_train += len(files)

folder = 'train_data/test'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
    files_validation += len(files)

print(files_train, files_validation)



# 数据扩增
train_datagen = ImageDataGenerator(
    # preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5)

test_datagen = ImageDataGenerator(
    # preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical",
    batch_size=batch_size
    )


# 可视化数据集
imgs, labels = next(train_generator)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        print(img.shape)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# plotImages(imgs)
# print(labels)


# 训练数据
model.fit(x=train_generator,
            steps_per_epoch=len(train_generator),
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            epochs=10,
            verbose=2
)

model.save('car2.h5')
print("save done...")
