import numpy as np
import math

"""Global Variables"""
n_neurons = 0
weights = []
biases = []
act_function = ""
act_value = 0
input = 0

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

activations_func = {
  'linear': linear,
  'sigmoid': sigmoid,
  'softmax': softmax,
  'relu': relu
}

def generate_layer(N_NEURONS, ACT_FUNCTION, WEIGHTS, BIASES):
  if(N_NEURONS < 1):
    raise ValueError("Neuron harus lebih dari 0")

  if(activations_func in ('linear', 'sigmoid', 'relu', 'softmax')):
    n_neurons = N_NEURONS
    act_function = activations_func[ACT_FUNCTION]
    weights = WEIGHTS
    biases = BIASES
    activation_value = None
  else:
    raise ValueError("Aktifasi fungsi harus salah satu dari 'linear', 'sigmoid', 'relu', 'softmax'")


def pass_forward(INPUT):
  input = INPUT
  act_value = act_function(INPUT)