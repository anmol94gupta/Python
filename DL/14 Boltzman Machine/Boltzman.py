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
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
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

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
# As RBM would need to tell whether the user liked it or not. Therefore converting the ratings to binary 0,1 did not like or like
# 0 changing to -1 as that was not a rating
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
# Classes are best for making whatever you want to make so always you classes and use the objects.
class RBM():
  def __init__(self, nv, nh): # number of visible nodes, number of hidden nodes
    self.W = torch.randn(nh, nv) # weights. Variables of the object are initialised with a self.
    # initializes a tensor of size nh and nv according to normal distribution with mean 0 and variance 1
    self.a = torch.randn(1, nh) # P(h/v)
    # 1 is to make a 2d tensor. 1 corresponds to the batch (first dimention) and nv to the bias
    self.b = torch.randn(1, nv) # P(v/h)
  def sample_h(self, x): # x = the visible neurons v in the probability P h given v
    wx = torch.mm(x, self.W.t()) # wieghts time visible neurons. mm for mulitplying 2 tensors. .t() to take transpose for multiplication
    activation = wx + self.a.expand_as(wx) # activation functions are linear, which is wx + bias. to add a new dimention to the bias we use expand function. so that bias is applied to each line of mini batch 
    p_h_given_v = torch.sigmoid(activation) # Probability that the hidden node is activated, given the value of the visible node.
    return p_h_given_v, torch.bernoulli(p_h_given_v) # samples of hidden neurons based on the probability
  def sample_v(self, y): # y = the hidden nodes
    wy = torch.mm(y, self.W) # transpose not needed. as W is weight matrix of P v given h 
    activation = wy + self.b.expand_as(wy)
    p_v_given_h = torch.sigmoid(activation) 
    return p_v_given_h, torch.bernoulli(p_v_given_h)
  def train(self, v0, vk, ph0, phk): # For contrastive divergence - for approximating the loglikelihood  gradient. Minimize the energy or maximize the log likelihood.
      # Direct comptation of gradient is heavy, therefore we try to approximate it.
      # Using Gibbs sampling 
      # v0 is input vector of one user
      # vk is visible nodes obtained after k iterations
      # ph0 vector of probabilities that at the first iterations that the hidden nodes = 1 given the values of v0 
      # phk vector of probabilities after k iterations that the hidden nodes = 1 given the values of vk
    # Updating weights - adding 
    self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
    self.b += torch.sum((v0 - vk), 0)
    self.a += torch.sum((ph0 - phk), 0)
    
    
nv = len(training_set[0])
nh = 100
batch_size = 100 # Can be tuned
rbm = RBM(nv, nh) # RBM model initiation

#Training RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0. # Float counter that will increase after each epoch
  # starting from 0 to nb_users - batch_size with jump of batch_size
  for id_user in range(0, nb_users - batch_size, batch_size):
    vk = training_set[id_user : id_user + batch_size]
    v0 = training_set[id_user : id_user + batch_size]
    ph0,_ = rbm.sample_h(v0) # ,_ to take only the first element of the function returns
    # Contrastive divergence
    for k in range(10):
      _,hk = rbm.sample_h(vk) # to get the values of the bernouli
      _,vk = rbm.sample_v(hk)
      vk[v0<0] = v0[v0<0]
    phk,_ = rbm.sample_h(vk)
    rbm.train(v0, vk, ph0, phk)
    train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
    s += 1.
  print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

#Testing RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])) 
        # Calculating loss mean of the absolute distance between actual and predicted
        s += 1.
print('test loss: '+str(test_loss/s))
