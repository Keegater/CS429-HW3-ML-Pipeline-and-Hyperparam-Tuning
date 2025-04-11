import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

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




# Show random samples from MNIST and Fashion data
plt.figure(figsize=(8, 5))

plt.subplot(1, 2, 1)
rand = np.random.choice(len(mnist_train_images))
plt.imshow(mnist_train_images[rand], cmap='gray')
plt.title(f"MNIST Label: {mnist_train_labels[rand]}")
plt.axis('off')

plt.subplot(1, 2, 2)
rand = np.random.choice(len(fashion_train_images))
plt.imshow(fashion_train_images[rand], cmap='gray')
plt.title(f"Fashion Label: {fashion_train_labels[rand]}")
plt.axis('off')

plt.show()


#####################################################################################
##################### part 3 ##################################################

param_grids = {
    'linear': {
        'svc__C': [0.0095, 0.00985, 0.0099, 0.00995]
    },
    'rbf': {
        'svc__C': [7, 7.5, 8, 8.5],
        'svc__gamma': [0.0025, 0.0029, 0.003, 0.0031]
    },
    'poly': {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [0.001, 0.01],
        'svc__degree': [2, 3, 4]
    }
}


def run_pipeline(X_train, y_train, X_test, y_test, method='pca', dims=[50], kernels=['linear']):
    for dim in dims:
        # Choose reducer
        if method == 'lda':
            n_classes = len(np.unique(y_train))
            dim = min(dim, n_classes - 1)  # LDA limit
            reducer = LDA(n_components=dim)
        elif method == 'pca':
            reducer = PCA(n_components=dim)


        for kernel in kernels:
            print(f"\n🔁 Running: {method.upper()} | dim = {dim} | kernel = {kernel}")

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('reducer', reducer),
                ('svc', SVC(kernel=kernel))
            ])

            grid = GridSearchCV(pipeline, param_grids[kernel], cv=3, n_jobs=-1, verbose=3)
            grid.fit(X_train, y_train)

            print("Best parameters:", grid.best_params_)
            y_pred = grid.predict(X_test)
            print("Classification Report:")
            print(classification_report(y_test, y_pred))




run_pipeline(mnist_train_images_flat, mnist_train_labels,
             mnist_test_images_flat, mnist_test_labels,
             method='pca', dims=[50, 100, 200], kernels=['linear', 'rbf'])


run_pipeline(fashion_train_images_flat, fashion_train_labels,
             fashion_test_images_flat, fashion_test_labels,
             method='lda', dims=[9], kernels=['linear', 'poly'])
# Dims set to 9
# dim = min(dim, n_classes - 1)  # LDA limit
# limits the number to 9 for any higher value.
# Should we run multiple dims < 9 for testing? I havent read enough on this section to know the requirements
# but for now only having 9 recudes redundant runs and saves runtime.


