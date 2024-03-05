import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging

EPOCHS = 20
BATCH_SIZE = 1

def main():
    tf.get_logger().setLevel(logging.ERROR)
    tf.random.set_seed(7)

    # Load training and test datasets
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Standardize
    mean = np.mean(train_images)
    stddev = np.std(train_images)
    train_images = (train_images - mean) / stddev
    test_images = (test_images - mean) / stddev

    # One-hot encode labels
    train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = keras.utils.to_categorical(test_labels, num_classes=10)
    
    initializer = keras.initializer.RandomUniform()

if __name__ == "__main__":
    main()
