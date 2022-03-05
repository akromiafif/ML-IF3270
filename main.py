import numpy as np
import math

"""Global Variables"""
n_neurons = 0
weights = []
biases = []
act_function = ""
act_value = 0
input = 0


# ALTERNATIVE ACTIVATION FUNCTIONS
def activation_func(nama, x):
  switcher = {
    "linear" : float(x),
    "sigmoid" : float(1/(1 + math.exp(-x))),
    "relu" : float(max(x,0)),
    "softmax" : float(np.exp(x) / np.sum(np.exp(x))),
  }
  return switcher.get(nama, "")

"""RELU Activation Function"""
def relu(x):
  return float(max(x,0))

"""Linear Activation Function"""
def linear(x):
  return float(x)

"""Sigmoid Activation Function"""
def sigmoid(x):
  return float(1/(1 + math.exp(-x)))

"""Softmax Activation Function"""
def softmax(x):
  return float(np.exp(x) / np.sum(np.exp(x)))

# Activation Function ENUM
activations_func = {
  'linear': linear,
  'sigmoid': sigmoid,
  'softmax': softmax,
  'relu': relu
}

"""Generate Layers Function"""
def generate_layer(N_NEURONS, ACT_FUNCTION, WEIGHTS, BIASES):
  if (N_NEURONS < 1):
    raise ValueError("Neuron harus lebih dari 0")

  if ACT_FUNCTION.lower() in activations_func.keys():
    n_neurons = N_NEURONS
    act_function = activations_func[ACT_FUNCTION]
    weights = WEIGHTS
    biases = BIASES
    activation_value = None
  else:
    raise ValueError("Aktifasi fungsi harus salah satu dari 'linear', 'sigmoid', 'relu', 'softmax'")


"""Forward Pass Function"""
def pass_forward(INPUT):
  input = INPUT
  act_value = act_function(INPUT)