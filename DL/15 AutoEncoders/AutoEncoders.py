#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:43:02 2021
@author: anmol
"""

# Datasets : http://files.grouplens.org/datasets/movielens/ml-100k.zip" "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn # Module of torch for neural networks
import torch.nn.parallel # For parallel computation
import torch.optim as optim # For optimizer
import torch.utils.data
from torch.autograd import Variable # Stochastic gradient descent 

# We won't be using this dataset.
# movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
# users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
# ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
# Last column in ratings is the timestamp

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int') # Converting the dataset to a numpy arrray
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int') # Converting the dataset to a numpy arrray

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))# :,0 means all rows of column 0
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in columns
# As the input format is observations in lines and features in columns
def convert(data):
  new_data = []
  for id_users in range(1, nb_users + 1): # List for each user
    id_movies = data[:, 1] [data[:, 0] == id_users] # Pick all rows of first column where user id in all rows = id_users
    id_ratings = data[:, 2] [data[:, 0] == id_users] # Similarlt pick all ratings
    ratings = np.zeros(nb_movies) # As we need a total list for all movies so generating a zeroes list
    ratings[id_movies - 1] = id_ratings # Replacing the zeroes with ratings based on the id of movies so changing 0 to rating for that 
    new_data.append(list(ratings)) # Appending to empty list 
  # print(new_data)
  return new_data

# Calling the above function on the training and testing data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting data to torch tensors - Just converting the numpy array to torch tensor
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

"""## Creating the architecture of the Neural Network"""
# Making class for stacked auto encoder. Classes written in Capitals 
class SAE(nn.Module):
    def __init__(self, ): # Defines architechture of the Auto Encoder
        super(SAE, self).__init__()
        # To get inheritence methods from the modules
        self.fc1 = nn.Linear(nb_movies, 20)
        # fc1 is an object of the linear class. Self repersents our object.
        # full connection between feature input vector and first layer of 20 features (found by trying multiple values)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        # Fourth full connection
        self.activation = nn.Sigmoid() # Can try rectifier function as well.
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x) # as final part of decoding, therefore activation function not used
        return x
sae = SAE()
criterion = nn.MSELoss() # criterion is an object of the class MSELoss from the nn module
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) 
# RMS prop gave better results than Adam in this case
# RMSprop is a class from optim module of torch library
# the arguments that are in the function above are based on research done by tutor
"""## Training the SAE"""

nb_epoch = 200
for epoch in range(1, nb_epoch + 1): # as we need to move over all the epochs
  train_loss = 0 # loss error
  s = 0. # used to calculate RMSE at the end which is a float. Number of users that rated atleast one movie
  for id_user in range(nb_users):# No +1 as we are working in range of indices 0 to 942
    input = Variable(training_set[id_user]).unsqueeze(0)
    # Variable - to create a new dimension with unsqueeze function to create a batch on single input vector
    #batch of input vector
    target = input.clone()
    if torch.sum(target.data > 0) > 0: # to optimise memory and see only those users who rated atleast one movie
        # Checking if sum of ratings > 0 is > 0 or not
      output = sae(input) # Vector of predictor ratings. sae is an object of the SAE class
      target.require_grad = False # we don't compute the gradient with respect to the target
      output[target == 0] = 0 # we don't want to deal with movies that were not rated.
      loss = criterion(output, target)
      mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)# number of movies/ number of movies with positive rating + 1e-10 for making sure that the denominator is not zero   
      loss.backward()# calling the backward method
      train_loss += np.sqrt(loss.data*mean_corrector)# loss object, data in the loss objeect multiplyed by mean collector which is the adjustment factor to calculate the relevant mean. root taken for adjusted square loss
      s += 1. # number of users who rated atleast one movie, therefore increasing s.
      optimizer.step() # step method of optimizer to apply the optimizer.
      # back decides the direction, optimizer decides the amount.
  print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s)) # average of trainloss

"""## Testing the SAE"""

test_loss = 0
s = 0.
for id_user in range(nb_users):
  input = Variable(training_set[id_user]).unsqueeze(0)
  target = Variable(test_set[id_user]).unsqueeze(0)
  if torch.sum(target.data > 0) > 0:
    output = sae(input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criterion(output, target)
    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
    test_loss += np.sqrt(loss.data*mean_corrector)
    s += 1.
print('test loss: '+str(test_loss/s)) # average test loss over all the users that gave atleast 1 non zero rating