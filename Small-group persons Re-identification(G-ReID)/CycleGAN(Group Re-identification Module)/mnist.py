import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers, optimizers, datasets

(x,y), (x_val,y_val) =  datasets.mnist.load_data()
x = 2*tf.convert_to_tensor(x,dtype = tf.float32)/255.-1
y = tf.convert_to_tensor(y,dtype = tf.int32)

y = tf.one_hot(y,depth = 10)
print(x.shape, y.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
train_dataset = train_dataset.batch(512)


model = keras.Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(10)
])

with tf.GradientTape() as tape:
    x = tf.reshape(x,(-1,28*28))
    out = model(x)
    y_onehot = tf.one_hot(y, depth =10)
    loss = tf.square(out - y_onehot)
    loss = tf.reduce_sum(loss)/ x.shape[0]
    grads = tape.gradient(loss,model.trainable_variables)
    optimizers.apply_gradients(zip(grads,model.trainable_variables))

