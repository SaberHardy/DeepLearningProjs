from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(units=4, activation='relu', input_shape=[3]),
    layers.Dense(units=3, activation='relu')
])

print(model.summary())
