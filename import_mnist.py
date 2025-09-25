import struct
import gzip
from pathlib import Path
import urllib.request
import os

def download_mnist():
    import urllib.request
    from pathlib import Path

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
        encoded = [0.0] * 10
        encoded[label] = 1.0
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
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"Training images: {len(x_train)}")
    print(f"Pixels per image: {len(x_train[0])}")
    print(f"Training labels: {len(y_train)}")
    print(f"Classes per label: {len(y_train[0])}")
    print(f"Test images: {len(x_test)}")
    print(f"Test labels: {len(y_test)}")
    print(f"First image (first 10 pixels): {x_train[0][:10]}")
    print(f"First label: {y_train[0]}")
    print(x_train)
