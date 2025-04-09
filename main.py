import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

# MNIST Dataset
mnist_train_images = idx2numpy.convert_from_file("data/train-images.idx3-ubyte")
mnist_train_labels = idx2numpy.convert_from_file("data/train-labels.idx1-ubyte")
mnist_test_images  = idx2numpy.convert_from_file("data/t10k-images.idx3-ubyte")
mnist_test_labels  = idx2numpy.convert_from_file("data/t10k-labels.idx1-ubyte")

# Fashion Dataset
fashion_train_images = idx2numpy.convert_from_file("data/fashion-train-images.idx3-ubyte")
fashion_train_labels = idx2numpy.convert_from_file("data/fashion-train-labels.idx1-ubyte")
fashion_test_images  = idx2numpy.convert_from_file("data/fashion-t10k-images.idx3-ubyte")
fashion_test_labels  = idx2numpy.convert_from_file("data/fashion-t10k-labels.idx1-ubyte")

# Flattening
mnist_train_images_flat = mnist_train_images.reshape(mnist_train_images.shape[0], 784)
mnist_test_images_flat  = mnist_test_images.reshape(mnist_test_images.shape[0], 784)
fashion_train_images_flat = fashion_train_images.reshape(fashion_train_images.shape[0], 784)
fashion_test_images_flat  = fashion_test_images.reshape(fashion_test_images.shape[0], 784)




# Show random samples from MNIST and Fashion data
plt.figure(figsize=(8, 5))

plt.subplot(1, 2, 1)
rand = np.random.choice(len(mnist_train_images))
plt.imshow(mnist_train_images[rand], cmap='gray')
plt.title(f"MNIST Label: {mnist_train_labels[0]}")
plt.axis('off')

plt.subplot(1, 2, 2)
rand = np.random.choice(len(fashion_train_images))
plt.imshow(fashion_train_images[rand], cmap='gray')
plt.title(f"Fashion Label: {fashion_train_labels[0]}")
plt.axis('off')

plt.show()