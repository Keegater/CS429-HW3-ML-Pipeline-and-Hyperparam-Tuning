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

# === Parameter Grids for GridSearch === #
param_grids = {
    'linear': {
        'svc__C': [0.01, 0.1, 1, 10, 100, 500, 1000, 5000]
    },
    'rbf': {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': [0.0001, 0.001, 0.01, 0.1]
    },
    'poly': {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [0.001, 0.01],
        'svc__degree': [2, 3, 4]
    }
}

# === Function to Build and Run Pipeline === #
def run_pipeline(X_train, y_train, X_test, y_test, method='pca', dims=[50], kernels=['linear']):
    for dim in dims:
        # Choose reducer
        if method == 'lda':
            n_classes = len(np.unique(y_train))
            dim = min(dim, n_classes - 1)  # LDA limit
            reducer = LDA(n_components=dim)
        elif method == 'pca':
            reducer = PCA(n_components=dim)
        else:
            raise ValueError("method must be 'pca' or 'lda'")

        # Try all kernels
        for kernel in kernels:
            print(f"\n🔁 Running: {method.upper()} | dim = {dim} | kernel = {kernel}")

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('reducer', reducer),
                ('svc', SVC(kernel=kernel))
            ])

            grid = GridSearchCV(pipeline, param_grids[kernel], cv=3, n_jobs=-1, verbose=1)
            grid.fit(X_train, y_train)

            print("✅ Best parameters:", grid.best_params_)
            y_pred = grid.predict(X_test)
            print("📊 Classification Report:")
            print(classification_report(y_test, y_pred))




#
# run_pipeline(mnist_train_images_flat, mnist_train_labels,
#              mnist_test_images_flat, mnist_test_labels,
#              method='pca', dims=[50, 100, 200], kernels=['linear', 'rbf'])
#
# # or try LDA on Fashion MNIST
# run_pipeline(fashion_train_images_flat, fashion_train_labels,
#              fashion_test_images_flat, fashion_test_labels,
#              method='lda', dims=[50, 100], kernels=['linear', 'poly'])