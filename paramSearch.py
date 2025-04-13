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

# grab 20% subset of training data
subset_size = int(0.2 * len(mnist_train_images_flat))
subset_idx = np.random.choice(len(mnist_train_images_flat), size=subset_size, replace=False)
X_train_subset = mnist_train_images_flat[subset_idx]
y_train_subset = mnist_train_labels[subset_idx]

reducer = PCA(n_components=50)


#________________Linear_____________
linear_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reducer', reducer),
    ('svc', SVC(kernel='linear'))
])

coarse_linear_grid = {
    'svc__C': np.logspace(-4, 3, num=8)  # [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# coarse search subset
grid_linear_coarse = GridSearchCV(linear_pipeline, coarse_linear_grid, cv=3, n_jobs=-1, verbose=3)
grid_linear_coarse.fit(X_train_subset, y_train_subset)
print("Linear Kernel Coarse Search Best Parameters:", grid_linear_coarse.best_params_)

# refine search
svcC_best_linear = grid_linear_coarse.best_params_['svc__C']
refined_linear_grid = {
    'svc__C': np.linspace(svcC_best_linear * 0.5, svcC_best_linear * 1.5, num=5),
}

grid_linear_refined = GridSearchCV(linear_pipeline, refined_linear_grid, cv=3, n_jobs=-1, verbose=3)
grid_linear_refined.fit(X_train_subset, y_train_subset)
print("Linear Kernel Refined Search Best Parameters:", grid_linear_refined.best_params_)

#___________________________________________________________________________________

#________________RBF_____________
rbf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reducer', reducer),
    ('svc', SVC(kernel='rbf'))
])

coarse_rbf_grid = {
    'svc__C': np.logspace(-4, 3, num=8),       # [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    'svc__gamma': np.logspace(-4, 3, num=8)     # [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# coarse search subset
grid_rbf_coarse = GridSearchCV(rbf_pipeline, coarse_rbf_grid, cv=3, n_jobs=-1, verbose=3)
grid_rbf_coarse.fit(X_train_subset, y_train_subset)
print("RBF Kernel Coarse Search Best Parameters:", grid_rbf_coarse.best_params_)

# refine search
svcC_best_rbf = grid_rbf_coarse.best_params_['svc__C']
gamma_best_rbf = grid_rbf_coarse.best_params_['svc__gamma']
refined_rbf_grid = {
    'svc__C': np.linspace(svcC_best_rbf * 0.5, svcC_best_rbf * 1.5, num=5),
    'svc__gamma': np.linspace(gamma_best_rbf * 0.5, gamma_best_rbf * 1.5, num=5)
}

grid_rbf_refined = GridSearchCV(rbf_pipeline, refined_rbf_grid, cv=3, n_jobs=-1, verbose=3)
grid_rbf_refined.fit(X_train_subset, y_train_subset)
print("RBF Kernel Refined Search Best Parameters:", grid_rbf_refined.best_params_)

#___________________________________________________________________________________


#________________POLY_____________
poly_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reducer', reducer),
    ('svc', SVC(kernel='poly'))
])

coarse_poly_grid = {
    'svc__C': np.logspace(-1, 2, num=4),  # [0.1, 1, 10, 100]
    'svc__gamma': np.logspace(-3, -1, num=3),  # [0.001, 0.01, 0.1]
    'svc__degree': [2, 3, 4, 5, 6, 7, 8, 9]
}

# coarse search subset
grid_poly_coarse = GridSearchCV(poly_pipeline, coarse_poly_grid, cv=3, n_jobs=-1, verbose=3)
grid_poly_coarse.fit(X_train_subset, y_train_subset)
print("Poly Kernel Coarse Search Best Parameters:", grid_poly_coarse.best_params_)

# refine search
svcC_best_poly = grid_poly_coarse.best_params_['svc__C']
gamma_best_poly = grid_poly_coarse.best_params_['svc__gamma']
best_degree_poly = grid_poly_coarse.best_params_['svc__degree']

# ensures low of 2 and max of 9, no duplicates
low = max(2, best_degree_poly - 1)
high = min(9, best_degree_poly + 1)
svc_degree = list(range(low, high + 1))
refined_poly_grid = {
    'svc__C': np.linspace(svcC_best_poly * 0.5, svcC_best_poly * 1.5, num=5),
    'svc__gamma': np.linspace(gamma_best_poly * 0.5, gamma_best_poly * 1.5, num=5),
    'svc__degree': svc_degree
}

grid_poly_refined = GridSearchCV(poly_pipeline, refined_poly_grid, cv=3, n_jobs=-1, verbose=3)
grid_poly_refined.fit(X_train_subset, y_train_subset)
print("Poly Kernel Refined Search Best Parameters:", grid_poly_refined.best_params_)

#___________________________________________________________________________________


