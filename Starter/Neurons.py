from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

concrete = pd.read_csv("../datasets/Concrete_Data_Yeh.csv")
print(concrete.head)
"""
   'cement',   'slag',    'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age', 'csMPa'
0   540.0        0.0     0.0         162.0           2.5              1040.0              676.0       28     79.99
1   540.0        0.0     0.0         162.0           2.5              1055.0              676.0       28     61.89
"""
target = [8]

# Define the model with hidden layers
# model = keras.Sequential([
#     layers.Dense(units=512, activation="relu", input_shape=target),
#     layers.Dense(units=512, activation="relu"),
#     layers.Dense(units=512, activation="relu"),
#     layers.Dense(1)  # This one is the output of the neurons
# ])

"""
Or using this method also is better
"""
model = keras.Sequential([
    layers.Dense(32, input_shape=target),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.Activation('relu'),
    layers.Dense(1),
])
print(model.score)
