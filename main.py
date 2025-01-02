import numpy as np
import matplotlib.pyplot as plt
from Perceptron import perceptron, brute_force, found_margin
from svm import svm_s
from Adaboost import adaboost, run_adaboost_multiple_times


def main(data_txt):

    """
    Part II - Perceptron algorithm

    This part of the code runs the Perceptron algorithm on different sets of flower data.

    Using the library - Perceptron:
    - perceptron function: 
        * return the weights vector (w)
        * return the number of mistakes (num_of_mistakes) 
    - compute_margin function:
        * return the true maximum margin (c)

    We perform classification on the following pairs of species:
    a. Setosa and Versicolor
    b. Setosa and Virginica
    c. Versicolor and Virginica

    """
    a = [5,10]
    b = [5,6]
    c = [2,14]

    print(found_margin(a,b,c))

    # Read and write data from file to np array
    setosa, versicolor, virginica = write_from_file(data_txt)

    selected_h, alpha, emp_errors, true_errors = adaboost(versicolor, virginica)

    print("Selected hypotheses weights:", alpha)
    print("Empirical errors:", emp_errors)
    print("True errors:", true_errors)

    # avg_empirical_errors, avg_true_errors = run_adaboost_multiple_times(versicolor, virginica)

    # print("Empirical errors:", avg_empirical_errors)
    # print("True errors:", avg_true_errors)


    print(svm_s(setosa, versicolor))
    # Perceptron on Setosa and Versicolor
    print("a - Perceptron on Setosa and Versicolor :")
    w, num_of_mistakes = perceptron(setosa, versicolor) 
    c, a, b, d = brute_force(setosa, versicolor) 
    
    plot_results(setosa, 'setosa', versicolor, 'versicolor', w, a, b, d, False, c)
    plot_results(setosa, 'setosa', versicolor, 'versicolor', w, a, b, d, True, c)
    print(f'Final weights vector : {w}')
    print(f'Number of mistakes : {num_of_mistakes}')
    print(f'True maximum margin : {c}')

    # Perceptron on Setosa and virginica
    print("b - Perceptron on Setosa and virginica :")
    w, num_of_mistakes = perceptron(setosa, virginica)
    c, a, b, d = brute_force(setosa, virginica)
    print(svm_s(setosa, virginica))

    plot_results(setosa, 'setosa', virginica, 'virginica', w, a, b, d, False, c)
    plot_results(setosa, 'setosa', virginica, 'virginica', w, a, b, d, True, c)
    print(f'Final weights vector : {w}')
    print(f'Number of mistakes : {num_of_mistakes}')
    print(f'True maximum margin : {c}')

    # # Perceptron on versicolor and virginica
    # print("c - Perceptron on versicolor and virginica :")
    # # w, num_of_mistakes = perceptron(versicolor, virginica)  ---->  Infinte loop, cannot be classified linearly
    # # c = brute_force(versicolor, virginica)
    
    # print("Run time error - Cannot be classified linearly")
    # plot_results(versicolor, 'versicolor', virginica, 'virginica', w)

    """
    Part III - Adaboost algorithm

    This part of the code runs the Perceptron algorithm on different sets of flower data.
    We perform classification on the following pairs of species:
    a. Setosa and Versicolor
    b. Setosa and Virginica
    c. Versicolor and Virginica
    
    """


def write_from_file(data_txt):
    setosa = []
    versicolor = []
    virginica = []
    with open(data_txt, 'r') as file:

        for line in file:
            columns = line.split()  
            if columns[4] == 'Iris-setosa':
                setosa.append([float(columns[1]), float(columns[2])])
            if columns[4] == 'Iris-versicolor':
                versicolor.append([float(columns[1]), float(columns[2])])
            if columns[4] == 'Iris-virginica':
                virginica.append([float(columns[1]), float(columns[2])])
    setosa = np.array(setosa)
    versicolor = np.array(versicolor)
    virginica = np.array(virginica)

    return setosa, versicolor, virginica

def plot_results(data_1, name_1, data_2, name_2, w, a, b, c, booli, margin):
    # Plot the data points
    if booli:
        plt.scatter(data_1[:, 0], data_1[:, 1], color='red', label=f'Class 1 {name_1}')
        plt.scatter(data_2[:, 0], data_2[:, 1], color='blue', label=f'Class 2 {name_2}')
    else:
        plt.scatter(a[0], a[1], color='red', label=f'Class 1 {name_1}')
        plt.scatter(b[0], b[1], color='red', label=f'Class 1 {name_1}')
        plt.scatter(c[0], c[1], color='blue', label=f'Class 2 {name_2}')
    # Calculate the decision boundary: w1*x + w2*y + w0 = 0
    # Let's assume w = [w1, w2], so the line is w1*x + w2*y = 0
    # => y = -(w1/w2) * x (this is the decision boundary)
    if name_1 == 'setosa':

        print(data_1.shape)
        
        x_vals = np.linspace(min(data_1[:, 0].min(), data_2[:, 0].min()), max(data_1[:, 0].max(), data_2[:, 0].max()))
        y_vals =-(w[0]/w[1]) * x_vals  # Solving for y in terms of x using the hyperplane equation
        x_vals1 = np.linspace(min(a[0], b[0]), max(a[0], b[0]))
        y_vals1 = np.linspace(min(a[1]+margin, b[1]+margin), max(a[1]+margin, b[1]+margin))  # Solving for y in terms of x using the hyperplane equation
        # y_vals1 = np.linspace(min(a[1], b[1]), max(a[1], b[1])) 
        plt.plot(x_vals1, y_vals1, color='green', label='Decision boundary (Perceptron)')
        plt.plot(x_vals, y_vals, color='black', label='Decision boundary (Perceptron)')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Perceptron: Decision Boundary')
    plt.show()

if __name__ == "__main__":

    DATA_PATH = 'iris.txt'  # Update with the path to your data
    main(DATA_PATH)
