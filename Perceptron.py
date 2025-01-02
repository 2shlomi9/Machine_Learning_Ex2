import numpy as np
import math


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def found_margin(a, b, c):
    # Found the vector ab
    vector = np.array([b[0] - a[0], b[1] - a[1]])
    # Found ortogonaly vector of ab
    perpendicular_vector = np.array([vector[1], -vector[0]])
    # Perpendicular vector length
    magnitude_vec = np.linalg.norm(perpendicular_vector)
    # Normalize the perpendicular vector
    normalized_perpendicular = perpendicular_vector / magnitude_vec
    # Shift point c relative to point a
    shifted_c = np.array([c[0] - a[0], c[1] - a[1]])
    # Found the projection of shifted c on the perpendicular vector (distance)
    distance_to_line = abs(np.dot(shifted_c, normalized_perpendicular))
    # The true margin
    margin = distance_to_line / 2

    return margin

def is_point_in_margin(data_1, data_2, a, b, margin):
    # Calculate the margin from point to the line defined by a and b
    vector = np.array([b[0] - a[0], b[1] - a[1]])
    perpendicular_vector = np.array([vector[1], -vector[0]])
    magnitude_vec = np.linalg.norm(perpendicular_vector)
    normalized_perpendicular = perpendicular_vector / magnitude_vec
    
    for i in range(len(data_1)):
        shifted_point = np.array([data_1[0] - a[0], data_1[1] - a[1]])
        distance_to_line = abs(np.dot(shifted_point, normalized_perpendicular))
    if distance_to_line < margin:
        return False
        
    for i in range(len(data_2)):
        shifted_point = np.array([data_2[0] - a[0], data_2[1] - a[1]])
        distance_to_line = abs(np.dot(shifted_point, normalized_perpendicular))
    if distance_to_line < margin:
        return False
    return True

# def is_point_in_margin(data_1, data_2, margin, a, b, c):
#     vector = np.array([b[0] - a[0], b[1] - a[1]])
#     w = np.array([vector[1], -vector[0]])
#     side = np.dot(c, w)

#     if side < 0 :
#         y1 = -np.full(data_1.shape[0], margin)  
#         y2 = np.full(data_2.shape[0], margin)
#     elif side > 0:
#         y1 = np.full(data_1.shape[0], margin)  
#         y2 = -np.full(data_2.shape[0], margin)  

#     n = data_1.shape[0] + data_2.shape[0]  # Number of samples

#     X = np.vstack((data_1, data_2))  # Merging the two sets
#     Y = np.concatenate((y1, y2))  # Merge labels

#     is_true_margin = True

#     for i in range(n):
#         if np.dot(w, X[i]) < margin and Y[i]:
#             is_true_margin = False
    
#     return is_true_margin




def brute_force(data_1,data_2):
    
    min_dist = 0
    tmp_dist = 0

    a = np.zeros(data_1.shape[1])
    b = np.zeros(data_1.shape[1])
    c = np.zeros(data_1.shape[1])
#euclidean_distance(data_1[i], data_2[k]) + euclidean_distance(data_1[j], data_2[k])
    # Try all combinations of 3 points such that two points belong to data_1 and one to data_2.
    for i in range(len(data_1)):
        for j in range(i+1, len(data_1)):
            for k in range(len(data_2)):
                tmp_dist = found_margin(data_1[i],data_1[j],data_2[k])
                # tmp_dist = euclidean_distance(data_1[i], data_2[k]) + euclidean_distance(data_1[j], data_2[k])
                is_true = is_point_in_margin(data_1, data_2,data_1[i],data_1[j], tmp_dist)
                if i==0 and j==1 and k==0:
                    min_dist = tmp_dist
                    a = data_1[i]
                    b = data_1[j]
                    c = data_2[k]
                elif tmp_dist < min_dist and tmp_dist != 0 and is_true:
                    min_dist = tmp_dist
                    a = data_1[i]
                    b = data_1[j]
                    c = data_2[k]

    # Try all combinations of 3 points such that two points belong to data_2 and one to data_1.
    for i in range(len(data_2)):
        for j in range(i+1, len(data_2)):
            for k in range(len(data_1)):
                tmp_dist = found_margin(data_2[i],data_2[j],data_1[k])
                is_true = is_point_in_margin(data_2, data_1, data_2[i],data_2[j], tmp_dist)
                # tmp_dist = euclidean_distance(data_2[i], data_1[k]) + euclidean_distance(data_2[j], data_1[k])
                if tmp_dist < min_dist and tmp_dist != 0 and is_true == False:
                    min_dist = tmp_dist
                    a = data_1[i]
                    b = data_1[j]
                    c = data_2[k]

    return found_margin(a, b, c), a,b,c


def perceptron(data_1, data_2):
    
    y1 = np.ones(data_1.shape[0])  # All labels 1
    y2 = -np.ones(data_2.shape[0])  # All labels -1

    n = data_1.shape[0] + data_2.shape[0]  # Number of samples

    X = np.vstack((data_1, data_2))  # Merging the two sets
    Y = np.concatenate((y1, y2))  # Merge labels
    w = np.zeros(data_1.shape[1])  # Initialize weight vector to zero

    count_mistakes = 0
    mistakes = True  # Variable to check if any mistakes are made in the epoch

    while mistakes:
        mistakes = False  # Reset mistakes flag at the start of each epoch

        for i in range(n):
            prediction = np.dot(w, X[i])  # Prediction for current sample

            if prediction <= 0 and Y[i] == 1:  # Misclassified positive example
                w += X[i]  # Update weights
                mistakes = True
                count_mistakes += 1
            elif prediction > 0 and Y[i] == -1:  # Misclassified negative example
                w -= X[i]  # Update weights
                mistakes = True
                count_mistakes += 1

        # If no mistakes were made during the whole epoch, stop
        if not mistakes:
            break

    return w, count_mistakes
