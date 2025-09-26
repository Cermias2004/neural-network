import struct
import gzip
import urllib.request
import os
import random
import math

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []

        for layer in range(len(layer_sizes) - 1):
            input_size = layer_sizes[layer]
            output_size = layer_sizes[layer + 1]

            self.weights.append(weight_init(input_size, output_size))
            self.biases.append(bias_init(output_size))


    def forward_pass(self, input_data):
        layers = len(self.weights)
        current_input = input_data
        for layer_idx in range(layers):
            z = self.weight_input_bias_calc(
                self.weights[layer_idx],
                current_input,
                self.biases[layer_idx]
            )

            if layer_idx == layers - 1:
                current_input = self.softmax(z)
            else:
                current_input = self.relu(z)
        return current_input

    def relu(self, output_vector):
        return [max(0, x) for x in output_vector]

    def softmax(self, output_vector):
        exp_values = [math.exp(x - max(output_vector)) for x in output_vector]
        return [x /sum(exp_values) for x in exp_values]

    def weight_input_bias_calc(self, weights, inputs, bias):
        wib_output = []
        for neuron_idx in range(len(weights)):
            neuron_sum = 0
            for input_idx in range(len(inputs)):
                neuron_sum += weights[neuron_idx][input_idx] * inputs[input_idx]
            wib_output.append(neuron_sum + bias[neuron_idx])
        return wib_output

def weight_init(input_size, output_size):
    return [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(output_size)]

def bias_init(output_size):
    return [random.uniform(-1, 1) for _ in range(output_size)]


def download_mnist():
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    data_dir = "mnist_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename} . . .")
            try:
                urllib.request.urlretrieve(base_url + filename, filepath)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                return None

    return data_dir

def load_mnist_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

        image_bytes = f.read()

        images = []
        pixels_per_image = rows*cols

        for i in range(num_images):
            start_idx = i * pixels_per_image
            end_idx = start_idx + pixels_per_image

            image_pixels = [byte / 255.0 for byte in image_bytes[start_idx:end_idx]]
            images.append(image_pixels)

        return images

def load_mnist_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))

        labels = []
        label_bytes = f.read()

        for label in range(num_labels):
            labels.append(label_bytes[label])
        return labels

def one_hot_encode(labels, num_classes=10):
    one_hot = []

    for label in labels:
        encoded = [0] * num_classes
        encoded[label] = 1
        one_hot.append(encoded)

    return one_hot

def load_mnist():
    data_dir = download_mnist()

    x_train = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))

    x_test = load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test = load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":

    network = NeuralNetwork([784, 128, 64, 10])

    x_train, y_train, x_test, y_test = load_mnist()
    print(f"random probs: {network.forward_pass(x_train[0])}")

