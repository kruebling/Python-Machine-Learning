# Data Preprocessing

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Use Mean of col for missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_x = LabelEncoder()
x[:, 0] = label_x.fit_transform(x[:, 0])
encoder = OneHotEncoder(categorical_features = [0])
x = encoder.fit_transform(x).toarray()
label_y = LabelEncoder()
y = label_y.fit_transform(y)

# Splitting data into Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)