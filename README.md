# Neural Network from Scratch (MNIST Classifier)

This project implements a **fully connected feedforward neural network** (FFNN) from scratch using only Python’s standard libraries.
It trains and evaluates a model on the **MNIST handwritten digits dataset** (0–9) without using machine learning frameworks like TensorFlow or PyTorch.

---

## 🧠 Features

* Pure Python implementation — no external ML libraries
* Support for:

    * Arbitrary layer sizes
    * ReLU and Softmax activations
    * Mini-batch training
    * Dropout regularization
    * Cross-entropy loss
* Batch-based gradient accumulation and weight updates
* Automatic MNIST dataset download and parsing (gzip + struct)

---

## 📂 Project Structure

```
├── FFNN.py                # Main script containing the NeuralNetwork class and MNIST loader
├── mnist_data/            # Folder where MNIST data will be downloaded
└── README.md              # This file
```

---

## ⚙️ Installation

This project uses only Python standard libraries.
You’ll need **Python 3.8+**.

### 1. Clone the repository

```bash
git clone https://github.com/Cermias2004/neural-network.git
cd neural-network
```

### 2. Run the script

```bash
python FFNN.py
```

The program will automatically:

* Download the MNIST dataset (if not already downloaded)
* Train the network for the specified number of epochs
* Print loss and accuracy per epoch

---

## 🧮 Mathematical Formulation

### 1. Forward Propagation

For each layer $( l )$:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

#### Activation Functions:

* **ReLU (Hidden Layers)**:

$$
a^{(l)} = \text{ReLU}(z^{(l)}) = \max(0, z^{(l)})
$$

* **Softmax (Output Layer)**:

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

#### Dropout Regularization:

$$
a^{(l)} = \frac{m^{(l)} \odot a^{(l)}}{1 - p}
$$

where ( $m^{(l)}$ ) is a binary mask sampled from $(\text{Bernoulli}(1-p))$ and $( p )$ is the dropout rate.

---

### 2. Loss Function

**Cross-Entropy Loss:**

$$
L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

where:

* $( C )$: number of classes (10 for MNIST)
* $( y_i )$: one-hot encoded true label
* $( \hat{y}_i )$: predicted probability from the Softmax layer

The network tracks average loss per epoch:

$$
\text{AvgLoss} = \frac{1}{N} \sum_{n=1}^{N} L_n
$$

---

### 3. Backpropagation

For output layer error:

$$
\delta^{(L)} = \hat{y} - y
$$

For hidden layers (using ReLU derivative):

$$
\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \text{ReLU}'(a^{(l)})
$$

where:

$$
\text{ReLU}'(x) =
\begin{cases}
1, & \text{if } x > 0 \
0, & \text{otherwise}
\end{cases}
$$

#### Gradients:

$$
\nabla W^{(l)} = \delta^{(l)} (a^{(l-1)})^T
$$

$$
\nabla b^{(l)} = \delta^{(l)}
$$

---

### 4. Weight and Bias Updates

Using batch gradient descent:

$$
W^{(l)} := W^{(l)} - \eta \cdot \frac{1}{m} \sum_{i=1}^{m} \nabla W^{(l)}*i
$$

$$
b^{(l)} := b^{(l)} - \eta \cdot \frac{1}{m} \sum*{i=1}^{m} \nabla b^{(l)}_i
$$

where:

* $( \eta )$: learning rate
* $( m )$: batch size

---

### 5. Weight Initialization

**He Initialization:**

$$
W^{(l)} \sim U\left(-\sqrt{\frac{2}{n_{\text{in}}}}, \sqrt{\frac{2}{n_{\text{in}}}}\right)
$$

This helps stabilize gradients in ReLU networks.


---

## 🔢 Example Training Output

```
Downloading train-images-idx3-ubyte.gz . . .
Downloaded train-images-idx3-ubyte.gz
Epoch 1/5 - Loss: 0.9251 - Accuracy: 0.8752
Epoch 2/5 - Loss: 0.6274 - Accuracy: 0.9028
Epoch 3/5 - Loss: 0.5137 - Accuracy: 0.9161
Epoch 4/5 - Loss: 0.4519 - Accuracy: 0.9235
Epoch 5/5 - Loss: 0.4032 - Accuracy: 0.9289
```

---

## 🧮 Key Functions

| Function           | Description                                         |
| ------------------ | --------------------------------------------------- |
| `forward_pass()`   | Runs inputs through all layers using ReLU + Softmax |
| `back_prop()`      | Computes error and gradients                        |
| `dropout()`        | Applies dropout regularization during training      |
| `update_weights()` | Updates weights using accumulated gradients         |
| `load_mnist()`     | Downloads and loads the MNIST dataset               |
| `predict()`        | Returns the predicted digit for a given input       |

---

## 🧰 Configuration

You can modify these parameters in the `__main__` block:

```python
network = NeuralNetwork([784, 128, 64, 10])
batch_size = 32
epochs = 5
```

You can also adjust:

* `learning_rate` in the constructor
* `dropout_rate` (default 0.2)

---

## 🧑‍💻 Author

Developed by **Christian Ermias**

A lightweight educational implementation for understanding neural networks without external libraries.

---

## 📜 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute it for educational purposes.

---
