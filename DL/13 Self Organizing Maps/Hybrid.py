#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 00:04:37 2021

@author: anmol
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Credit_Card_Data.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# SOM diagram
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding and Printing frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(7,5)], mappings[(1,8)]), axis = 0)
#frauds = mappings[(8,8)]
frauds = sc.inverse_transform(frauds)

print("Fraud Customer's")
for i in frauds[:, 0]:
  print(int(i))


customers = dataset.iloc[:, 1:].values
# all columns index 1 and forward

is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
  if dataset.iloc[i,0] in frauds:
    is_fraud[i] = 1
    
# print(is_fraud)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

################

# ANN

import tensorflow as tf
tf.__version__

ann = tf.keras.models.Sequential()

#Input layer
ann.add(tf.keras.layers.Dense(units=2, activation='relu')) #as dataset is very simple

#Output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Training 
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#To training set
ann.fit(customers, is_fraud, batch_size = 1, epochs = 5)

#Predicting
y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()] # Sorting by column 1 like spreadsheets


