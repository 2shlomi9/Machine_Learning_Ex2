from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

def svm_s(data_1, data_2):
    """
    SVM with linear kernel for two datasets.

    Parameters:
    data_1: numpy array of shape (n_samples1, n_features) - First class
    data_2: numpy array of shape (n_samples2, n_features) - Second class

    Returns:
    clf: Trained SVM model
    """
    # Labels for the two classes
    y1 = np.ones(data_1.shape[0])  # Class 1 labels
    y2 = -np.ones(data_2.shape[0])  # Class 2 labels

    # Combine the data and labels
    X = np.vstack((data_1, data_2))
    Y = np.concatenate((y1, y2))

    # Train SVM with linear kernel
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, Y)

    # Plot decision boundary and margin
    plt.figure(figsize=(8, 6))

    # Scatter plot of the two datasets
    plt.scatter(data_1[:, 0], data_1[:, 1], color='red', label='Class 1', s=30)
    plt.scatter(data_2[:, 0], data_2[:, 1], color='blue', label='Class 2', s=30)

    # Get separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1)
    yy = a * xx - clf.intercept_[0] / w[1]

    # Plot the separating hyperplane
    plt.plot(xx, yy, 'k-', label='Decision boundary')

    # Plot margin
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin
    plt.plot(xx, yy_down, 'k--', label='Margin')
    plt.plot(xx, yy_up, 'k--')

    # Highlight support vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], 
                s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.legend()
    plt.grid()
    plt.show()

    return margin
