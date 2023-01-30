import tensorflow as tf 

print("num GPUs Available :",len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.__version__)
