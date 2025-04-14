import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

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


"""
To save runtime, hyperparam array ranges were "trimmed" by performing a coarse search on an initial log range of 8 values. 
Testing was done using 20% of the dataset. The best found hyperparams were then tested again (with 20% of data)
on a range of 5 values linearly spaced between 0.5X and 1.5X the previous best found value. The final "refined" values
are then used to build a linearly spaced array of 5 values 0.5X and 1.5X. These are finally tested on the whole dataset. 

This eliminates extreme parameters which may heavily impact runtime and allows us to choose the values for final testing
more precisely. 
"""
refined_svcC_best_linear = 0.0125

refined_svcC_best_rbf = 12.5
refined_gamma_best_rbf = 0.0015

refined_svcC_best_poly = 0.05
refined_gamma_best_poly = 0.05
refined_degree_best_poly = 3

# build degree array, ensure max of 9 and min of 2 with no duplicates
low = max(2, refined_degree_best_poly - 1)
high = min(9, refined_degree_best_poly + 1)
svc_degree_final_poly = list(range(low, high + 1))


##################### part 3 ##################################################
# build hyperparam arrays of 5 values within range of 0.5X and 1.5X of refined value
param_grids = {
    'linear': {
        'svc__C': np.linspace(refined_svcC_best_linear * 0.75, refined_svcC_best_linear * 1.5, num=4)
    },
    'rbf': {
        'svc__C': np.linspace(refined_svcC_best_rbf * 0.75, refined_svcC_best_rbf * 1.5, num=4),
        'svc__gamma': np.linspace(refined_gamma_best_rbf * 0.75, refined_gamma_best_rbf * 1.5, num=4)
    },
    'poly': {
        'svc__C': np.linspace(refined_svcC_best_poly * 0.75, refined_svcC_best_poly * 1.5, num=4),
        'svc__gamma': np.linspace(refined_gamma_best_poly * 0.75, refined_gamma_best_poly * 1.5, num=4),
        'svc__degree': svc_degree_final_poly
    }
}

results = []

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

            grid = GridSearchCV(pipeline, param_grids[kernel], cv=3, n_jobs=-1, verbose=1)
            grid.fit(X_train, y_train)

            print("Best parameters:", grid.best_params_)
            y_pred = grid.predict(X_test)
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

            # generate and print confusion matrices
            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)
            plot_confusion_matrix(cm, f"Confusion Matrix for: {kernel} SVC ({method.upper()}, dim = {dim})")

            accuracy = grid.score(X_test, y_test)
            best_params = grid.best_params_
            results.append({
                'Reducer': method.upper(),
                'Dim': dim,
                'Kernel': kernel,
                'Accuracy': accuracy,
                'BestParams': best_params,
                'Confusion Matrix': cm,
            })


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()



run_pipeline(mnist_train_images_flat, mnist_train_labels,
             mnist_test_images_flat, mnist_test_labels,
             method='pca', dims=[50, 100, 200], kernels=['linear', 'rbf', 'poly'])

run_pipeline(mnist_train_images_flat, mnist_train_labels,
             mnist_test_images_flat, mnist_test_labels,
             method='lda', dims=[9], kernels=['linear', 'rbf', 'poly'])

run_pipeline(fashion_train_images_flat, fashion_train_labels,
             fashion_test_images_flat, fashion_test_labels,
             method='pca', dims=[50, 100, 200], kernels=['linear', 'rbf', 'poly'])

run_pipeline(fashion_train_images_flat, fashion_train_labels,
             fashion_test_images_flat, fashion_test_labels,
             method='lda', dims=[9], kernels=['linear', 'rbf', 'poly'])
# Dims set to 9
# dim = min(dim, n_classes - 1)  # LDA limit
# limits the number to 9 for any higher value.

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("results.csv", index=False)




