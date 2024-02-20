import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow import keras
from tensorflow.keras import layers

red_wine = pd.read_csv('../datasets/winequality-red.csv')

# Create training and validation split
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)

x_train = df_train.drop('quality', axis=1)
x_valid = df_train.drop('quality', axis=1)

y_train = df_train['quality']
y_valid = df_valid['quality']

print(df_train)


