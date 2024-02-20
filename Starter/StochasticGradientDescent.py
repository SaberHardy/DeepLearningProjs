import pandas as pd
from IPython.display import display

red_wine = pd.read_csv('../datasets/winequality-red.csv')

# Create training and validation split
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)


# display(df_train.head())
# print("================================")
# display(df_valid.head())

# Scale the data into [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)

df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)


x_train = df_train.drop('quality', axis=1)
x_valid = df_train.drop('quality', axis=1)

y_train = df_train['quality']
y_valid = df_valid['quality']

print(x_train.shape)


