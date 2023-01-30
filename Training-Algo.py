import tensorflow as tf
from keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

#data collection and processing :

IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS =3
EPOCHS=25

#replace the paths down below with the paths of your local dataset :

dataset =tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\lazra\OneDrive\Bureau\PPP-V3\Apple\train",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
valid_ds=tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\lazra\OneDrive\Bureau\PPP-V3\Apple\valid",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names=dataset.class_names

train_size = 0.9
test_size=0.1
valid_size=1

train_ds=dataset.take(436)
test_ds=dataset.skip(436)


train_ds= train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds= test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
valid_ds= valid_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


resize_and_rescale= tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255)
])

#Model building :


input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes=4


model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(n_classes,activation='softmax'),
])


model.build(input_shape=input_shape)


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


history=model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=valid_ds
)

#evaluating my model :

Score=model.evaluate(test_ds)
print("\n")
print(Score)


#vizualizing the data :

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), acc, label="Training accuracy")
plt.plot(range(EPOCHS), val_acc, label="Validation accuracy")
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(range(EPOCHS), loss, label="Training loss")
plt.plot(range(EPOCHS), val_loss, label="Validation loss")
plt.legend(loc='upper right')
plt.title('Training and Validation loss')
plt.show()



#Model Saving : 

#you can save the model wherever you want , you just have to specify the path down below :
model_version=1
model.save(r"C:\Users\lazra\OneDrive\Bureau\PPP-V3\models\1")
print("module saved")

