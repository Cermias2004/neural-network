import struct
import gzip
import urllib.request
import os
import random
import math

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1, dropout_rate=0.2):
        self.errors = []
        self.loss = []
        self.current_inputs = []
        self.weights = []
        self.biases = []
        self.gradients = []
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        for layer in range(len(layer_sizes) - 1):
            input_size = layer_sizes[layer]
            output_size = layer_sizes[layer + 1]

            self.weights.append(weight_init(input_size, output_size))
            self.biases.append(bias_init(output_size))

    def predict(self, inputs):
        self.forward_pass(inputs)
        return self.current_inputs[-1].index(max(self.current_inputs[-1]))

    def get_weights(self):
        return self.weights

    def get_gradients(self):
        return self.gradients

    def get_errors(self):
        return self.errors

    def get_inputs(self):
        return self.current_inputs

    def get_biases(self):
        return self.biases

    def get_learning_rate(self):
        return self.learning_rate

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def get_loss(self):
        return self.loss

    def avg_loss(self):
        return sum(self.loss) / len(self.loss)

    def dropout(self, vector, drop_prob):
        if drop_prob<=0.0:
            return vector
        mask = [0 if random.random() <= drop_prob else 1 for _ in vector]
        return [v*m / (1-drop_prob) for v,m in zip(vector,mask)]

    def back_prop(self, one_hot):
        self.calc_error(one_hot)
        self.gradient()
        self.update_weights()
        self.update_biases()

    def update_weights(self):
        for layer in range(len(self.weights)):
            for i in range(len(self.weights[layer])):
                for j in range(len(self.weights[layer][i])):
                    self.weights[layer][i][j] -= self.learning_rate * self.gradients[(len(self.weights) - 1) - layer][i][j]

    def update_biases(self):
        for layer in range(len(self.biases)):
            for i in range(len(self.biases[layer])):
                self.biases[layer][i] -= self.learning_rate * self.errors[layer + 1][i]

    def gradient(self):
        self.gradients = []
        for layer in range(len(self.errors) - 1, -1, -1):
            layer_grads = []
            for neuron in range(len(self.errors[layer])):
                neuron_grads = []
                for input in range(len(self.current_inputs[layer - 1])):
                    neuron_grads.append(self.errors[layer][neuron] * self.current_inputs[layer - 1][input])
                layer_grads.append(neuron_grads)
            self.gradients.append(layer_grads)

    def calc_error(self, one_hot):
        layers = len(self.current_inputs) - 1
        self.errors = [None] * (layers + 1)

        self.errors[layers] = [
            self.current_inputs[layers][i] - one_hot[i] for i in range(len(self.current_inputs[layers]))
        ]

        self.loss.append(-math.log(sum(self.current_inputs[layers][i] * one_hot[i] for i in range(len(self.current_inputs[layers]))) + 1e-15))

        for layer_idx in range(layers - 1, -1, -1):
            prev_weight = self.weights[layer_idx]
            prev_error = self.errors[layer_idx + 1]
            new_error = []
            for j in range(len(self.current_inputs[layer_idx])):
                weighted_sum = sum(prev_weight[k][j] * prev_error[k] for k in range(len(prev_error)))
                new_error.append(weighted_sum * self.relu_derivative(self.current_inputs[layer_idx][j]))
            self.errors[layer_idx] = new_error

    def forward_pass(self, input_data, train=True):
        layers = len(self.weights)
        self.current_inputs = [input_data]
        for layer_idx in range(layers):
            z = self.weight_input_bias_calc(
                self.weights[layer_idx],
                self.current_inputs[layer_idx],
                self.biases[layer_idx]
            )

            if layer_idx == layers - 1:
                self.current_inputs.append(self.softmax(z))
            else:
                activated = self.relu(z)
                if train:
                    activated = self.dropout(activated, self.dropout_rate)
                self.current_inputs.append(activated)

    def relu_derivative(self, x):
        return 1 if x > 0 else 0

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
    limit = math.sqrt(2 / input_size)
    return [[random.uniform(-limit, limit) for _ in range(input_size)] for _ in range(output_size)]


def bias_init(output_size):
    return [0.1 for _ in range(output_size)]


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

    batch_size = 32
    epochs = 5

    for epoch in range(epochs):
        combine = list(zip(x_train, y_train))
        random.shuffle(combine)
        x_train, y_train = zip(*combine)

        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            for x,y in zip(x_batch,y_batch):
                network.forward_pass(x)
                network.back_prop(y)

        correct = 0
        for x,y in zip(x_test, y_test):
            pred = network.predict(x)
            if y[pred] == 1:
                correct += 1
        acc = correct / len(x_test)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {network.avg_loss():.4f} - Accuracy: {acc:.4f}")