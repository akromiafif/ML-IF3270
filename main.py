from asyncio.windows_events import NULL
import numpy as np
import math
import json

''' --- Activation Function --- '''
def activation_func(nama, x):
  switcher = {
    "linear" : x,
    "sigmoid" : 1 / (1 + np.exp(-x)),
    "relu" : np.maximum(0, x),
    "softmax" : np.exp(x) / np.sum(np.exp(x)),
  }
  return switcher.get(nama, "")

''' --- Info --- '''
def print_info():
    print(f"# Jumlah Layer: {len(layers)}")
    for idx in range(len(layers)):
        print(f"Layer ({idx+1})")
        print(f"- Jumlah Neuron: {layers[idx]['n_neuron']}")
        print(f"- Jenis Fungsi Aktivasi: {layers[idx]['activation_function']}")
        print(f"- Nilai Aktivasi: \n\t{layers[idx]['activation_value']}")
        print(f"- Bobot: \n\t{layers[idx]['weights']}")
        print(f"- Bias: \n\t{layers[idx]['biases']}\n")
    print(f"# Hasil Prediksi: {final_predictions}")

''' --- File Handler --- '''
def json_to_data(filename):
    f = open(filename)
    model = json.load(f)
    f.close()

    return model

''' --- Predict --- '''
def predict(array):

    prediction = NULL
    for idx in range(len(layers)):
        x = np.dot(array, np.array(layers[idx]["weights"])) + np.array(layers[idx]["biases"]) \
            if idx == 0 else np.dot(layers[idx-1]["activation_value"], np.array(layer["weights"])) + np.array(layer["biases"])

        layers[idx]["activation_value"] = activation_func(layers[idx]["activation_function"], x)
    
    last_layer_activation_value = layers[-1]["activation_value"]
    prediction = np.copy(last_layer_activation_value)
    prediction = prediction.reshape(prediction.shape[0], 1)

    return prediction

def final_predict(predictions):
    for prediction in predictions:
        final_predictions.append(0 if prediction < 0.5 else 1)

''' --- Main Program ---'''
FILENAME = 'model.json'
ACTIVATION_FUNCTION = ["linear", "sigmoid", "relu", "softmax"]

layers = []
final_predictions = []

if __name__ == "__main__":
    model = json_to_data(FILENAME)
    print(f"\nModel: \n{model}\n")
    for layer in model:

        # Validate
        if layer["n_neuron"] < 1:
            ValueError("Neuron harus lebih dari 1")
        if layer["activation_function"].lower() not in ACTIVATION_FUNCTION:
            raise ValueError("Fungsi Aktivasi harus salah satu dari 'linear', 'sigmoid', 'relu', 'softmax'")

        layers.append(layer)

    predictions = predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    final_predict(predictions)

    print_info()