from fileinput import filename

from black import out
import numpy as np
import math

'''Initizialization'''
weights = []
active_func = []
num_layer = 0
output = -1

"""RELU Activation Function"""
def relu(x):
  return float(max(x,0))

"""Linear Activation Function"""
def linear(x):
  return float(x)

"""Sigmoid Activation Function"""
def sigmoid(x):
  return 1/(1+math.exp(-x))

"""Softmax Activation Function"""
def softmax(x):
  return np.exp(x) / np.sum(np.exp(x))

def openFile(filename):

  file = open(filename)
  num_layer = int(file.readline())
    
  for i in range(num_layer):
      width, act = file.readline().split()
      width = int(width)
      active_func.append(act)
      
      weightMatrix = []
      for j in range(width):
          weightMatrix.append([int(x) for x in file.readline().split()])
      
      weights.append(np.array(weightMatrix))

  # Init output
  output = np.NaN
  print(output)

def predict(inp: np.array):
    enter = inp
    for i in range(num_layer):
      w = weights[i].transpose()

      # Melakukan perkalian matriks
      net = enter.dot(w)

      # Fungsi aktivasi
      vf = np.NaN
      raw = np.NaN
      if (active_func[i] == "relu"):
        vf = np.vectorize(relu)
        raw = vf(net)
      elif (active_func[i] == "linear"):
        vf = np.vectorize(linear)
        raw = vf(net)
      elif (active_func[i] == "sigmoid"):
        vf = np.vectorize(sigmoid)
        raw = vf(net)
      elif (active_func[i] == "softmax"):
        raw = softmax(net)
      else: 
        raise Exception("The", active_func[i], "is not available in this model.")

      # Jika belum ke output layer, maka kasih bias
      if (i < num_layer - 1):
        enter = np.array([[1 for i in range(inp.shape[0])]])
        trns = raw.transpose()
        enter = np.vstack((enter,trns)).transpose()
      else:
        enter = raw
    output = enter
    print(output)
    return output


if __name__ == "__main__":
  openFile('tester.txt')
  print(weights)
  #predict(np.array([[1,1,1]]))