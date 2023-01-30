import tensorflow as tf
from keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

#importing our Built Model :

#put the path of your model between parentheses :

model=tf.keras.models.load_model(r"C:\Users\lazra\OneDrive\Bureau\PPP-V3\models\1")


#Defining Variables : 

IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS =3
EPOCHS=50

#Importing Databases :
 
#put the path of your dataset between parentheses :

dataset =tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\lazra\OneDrive\Bureau\PPP-V3\Apple\train",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names=dataset.class_names


#Defining our testing Dataset +Caching & Chuffling :

test_ds=dataset.skip(436)
test_ds= test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


#evaluating my model :

Score=model.evaluate(test_ds)
print("\n")
print(Score)


#Predicting which plant disease from the test_ds : 
def predict(model, img):
    img_array=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array=tf.expand_dims(img_array, 0)#create a batch

    predictions=model.predict(img_array)
    predicted_class=class_names[np.argmax(predictions[0])]
    confidence=round(100*(np.max(predictions[0])),2)
    return predicted_class, confidence

plt.figure(figsize=(15,15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))

        predicted_class,confidence=predict(model,images[i].numpy())
        actual_class=class_names[labels[i]]
        plt.title(f"Actual:{actual_class},\n predicted:{predicted_class},\n Confidence:{confidence}%")
        plt.axis("off")
plt.show()