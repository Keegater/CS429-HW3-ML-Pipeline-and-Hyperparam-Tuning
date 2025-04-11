import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # You could also use LDA if desired
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings
import idx2numpy


warnings.filterwarnings("ignore")

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


# For this example, we use PCA with 50 components.
reducer = PCA(n_components=50)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reducer', reducer),
    ('svc', SVC(kernel='poly'))
])


coarse_param_grid = {
#--------------------- RBF -----------------------------
    # 'svc__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], # Best: 100
    # 'svc__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] # Best: 0.001
#---------------------
    # 'svc__C': [0.1, 1, 5, 10, 50, 80, 100, 200, 500], # Best: 5
    # 'svc__gamma': [0.0001, 0.0005, 0.0008, 0.001, 0.0012, 0.002, 0.005, 0.01] #Best: 0.002
#---------------------
    # 'svc__C': [2, 3, 4, 4.5, 5, 5.5, 6, 7, 8],  # Best: 8
    # 'svc__gamma': [0.001, 0.0015, 0.00175, 0.002, 0.00225, 0.00250, 0.003, 0.004]  # Best: 0.003
#--------------------------------------------------------

#--------------------- Linear -----------------------------
    # 'svc__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Best: 0.01
#---------------------
    # 'svc__C': [0.001, 0.005, 0.009, 0.0095, 0.01, 0.015, 0.02, 0.05, 0.1],  # Best: 0.01
#---------------------
    # 'svc__C': [ 0.0095, 0.0099, 0.00995, 0.00999, 0.01, 0.0101, 0.0105, 0.0101],  # Best: 0.0099
#--------------------------------------------------------

#--------------------- Poly -----------------------------
    'svc__C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1],
    'svc__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'svc__degree': [2, 3, 4, 5, 6, 7, 8, 9]



}

# Select a random 20% subset of the training data
subset_size = int(0.2 * len(mnist_train_images_flat))
subset_idx = np.random.choice(len(mnist_train_images_flat), size=subset_size, replace=False)
X_train_subset = mnist_train_images_flat[subset_idx]
y_train_subset = mnist_train_labels[subset_idx]


# Run the coarse grid search on the subset
grid_coarse = GridSearchCV(pipeline, coarse_param_grid, cv=3, n_jobs=-1, verbose=3)
grid_coarse.fit(X_train_subset, y_train_subset)

print("Coarse Search Best Parameters:", grid_coarse.best_params_)
