import numpy as np
import math

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
