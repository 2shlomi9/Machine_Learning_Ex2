import numpy as np
import random

def generate_hypothesis_set(data_1, data_2):
    """
    Generate a set of weak classifiers (lines) defined by every pair of points.
    """
    hypotheses = []

    for i in range(len(data_1)):
        for j in range(len(data_2)):
            p1, p2 = data_1[i], data_2[j]

            def weak_classifier(x, p1=p1, p2=p2):
                """
                Classify based on the position relative to the line defined by p1 and p2.
                """
                # Check which side of the line the point lies on
                return 1 if (x[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (x[0] - p1[0]) else -1

            hypotheses.append(weak_classifier)

    return hypotheses

def adaboost(data_1, data_2, k=8):
    """
    Perform AdaBoost using a hypothesis set of weak classifiers defined by data_1 and data_2.
    """
    # Combine data and assign labels
    data = np.vstack([data_1, data_2])
    labels = np.hstack([np.ones(len(data_1)), -np.ones(len(data_2))])

    # Split into train and test
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_indices = indices[:len(data) // 2]
    test_indices = indices[len(data) // 2:]

    train_data, train_labels = data[train_indices], labels[train_indices]
    test_data, test_labels = data[test_indices], labels[test_indices]

    # Generate hypothesis set
    hypotheses = generate_hypothesis_set(train_data, train_data)

    # Initialize weights
    n_train = len(train_data)
    D = np.ones(n_train) / n_train

    selected_hypotheses = []
    alpha = []

    for t in range(k):
        # Calculate weighted error for each hypothesis
        errors = []
        for h in hypotheses: 
            predictions = np.array([h(x) for x in train_data])
            weighted_error = np.sum(D * (predictions != train_labels))
            errors.append(weighted_error)

        # Select the hypothesis with minimum error
        best_h_idx = np.argmin(errors)
        best_h = hypotheses[best_h_idx]
        min_error = errors[best_h_idx]

        # Compute alpha
        alpha_t = 0.5 * np.log((1 - min_error) / max(min_error, 1e-10))
        alpha.append(alpha_t)
        selected_hypotheses.append(best_h)

        # Update weights
        predictions = np.array([best_h(x) for x in train_data])
        D = D * np.exp(-alpha_t * train_labels * predictions)
        D /= np.sum(D)  # Normalize

    # Compute empirical and true error for each k
    empirical_errors = []
    true_errors = []

    for t in range(1, k + 1):
        def H(x):
            return np.sign(sum(alpha[i] * selected_hypotheses[i](x) for i in range(t)))

        emp_error = np.mean([H(x) != y for x, y in zip(train_data, train_labels)])
        true_error = np.mean([H(x) != y for x, y in zip(test_data, test_labels)])

        empirical_errors.append(emp_error)
        true_errors.append(true_error)

    return selected_hypotheses, alpha, empirical_errors, true_errors

# Run AdaBoost multiple times and compute the average errors
def run_adaboost_multiple_times(data_1, data_2, num_runs=100):
    all_empirical_errors = []
    all_true_errors = []

    for _ in range(num_runs):
        _, _, empirical_errors, true_errors = adaboost(data_1, data_2)

        all_empirical_errors.append(empirical_errors)
        all_true_errors.append(true_errors)

    # Average errors over the 100 runs
    avg_empirical_errors = np.mean(all_empirical_errors, axis=0)
    avg_true_errors = np.mean(all_true_errors, axis=0)

    return avg_empirical_errors, avg_true_errors
