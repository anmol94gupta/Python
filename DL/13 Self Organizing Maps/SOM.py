#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 23:15:02 2021

@author: anmol
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Credit_Card_Data.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

print (X)

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#Plotting a SOM
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
# .T for transpose
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         #0 = cricle = did not get approval, 1 = square = get approval. Similarly picking up color
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 8,
         markeredgewidth = 2)
show()

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5,6)], mappings[(6,2)]), axis = 0) 
#Coordinates come from the above figure.
#concatenate the list of customers (2 lists and axis[0 as we want one below other vertically])

#frauds = sc.inverse_transform(frauds)

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))