from fileinput import filename

from black import out
import numpy as np
import math

"""Global Variables"""
n_neurons = 0
weights = []
biases = []
act_function = ""
act_value = 0
num_layer = 0
input = 0

def activation_func(nama, x):
  switcher = {
    # """Linear Activation Function""" #
    "linear" : float(x),

    # """Sigmoid Activation Function""" #
    "sigmoid" : float(1/(1 + math.exp(-x))),

    # """RELU Activation Function""" #
    "relu" : float(max(x,0)), 

    #"""Softmax Activation Function""" #
    "softmax" : float(np.exp(x) / np.sum(np.exp(x))),
  }

  if nama not in switcher.keys():
    raise ValueError("Aktifasi fungsi harus salah satu dari 'linear', 'sigmoid', 'relu', 'softmax'")
  else:
    return switcher.get(nama)

def openFile(filename):

  file = open(filename)
  num_layer = int(file.readline())
    
  for i in range(num_layer):
    width, act = file.readline().split()
    width = int(width)
    act_function.append(act)
    
    weightMatrix = []
    for j in range(width):
        weightMatrix.append([int(x) for x in file.readline().split()])
    
    weights.append(np.array(weightMatrix))

"""Generate Layers Function"""
def generate_layer(N_NEURONS, ACT_FUNCTION, WEIGHTS, BIASES, X):
  if (N_NEURONS < 1):
    raise ValueError("Neuron harus lebih dari 0")

  n_neurons = N_NEURONS
  act_function = activation_func(ACT_FUNCTION, X)
  weights = WEIGHTS
  biases = BIASES
  activation_value = None

"""Forward Pass Function"""
def pass_forward(INPUT):
  input = INPUT
  act_value = act_function(INPUT)

if __name__ == "__main__":
  print(activation_func('linear', 10))
 