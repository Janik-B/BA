import numpy as np


def gd(gradient, x_0, learning_rate=0.5, precision=0.001, max_iterations=1000):
    cur_x = x_0
    iteration = 0
    gradient_value = gradient(cur_x)
    norm_of_gradient = np.linalg.norm(gradient_value)
    x_values = [cur_x]
    while norm_of_gradient > precision and iteration < max_iterations:
        cur_x = cur_x - learning_rate * gradient_value
        iteration = iteration + 1
        x_values.append(cur_x)
        gradient_value = gradient(cur_x)
        norm_of_gradient = np.linalg.norm(gradient_value)

    print('Gradient descent took ', iteration, ' steps')
    return np.array(x_values)


def gradient_descent_momentum(gradient, x_0, learning_rate=0.5, precision=0.001, gamma=0.8,
                              max_iterations=1000):
    velocity = 0
    cur_x = x_0
    iteration = 0
    gradient_value = gradient(cur_x)
    norm_of_gradient = np.linalg.norm(gradient_value)
    x_values = [cur_x]
    while norm_of_gradient > precision and iteration < max_iterations:
        velocity = velocity * gamma - learning_rate * gradient_value
        cur_x = cur_x + velocity
        iteration = iteration + 1
        x_values.append(cur_x)
        gradient_value = gradient(cur_x)
        norm_of_gradient = np.linalg.norm(gradient_value)

    print('Gradient momentum descent took ', iteration, ' steps')
    return np.array(x_values)


def cgd(gradient, x_0, gamma=1, eta=0.5, precision=0.001, max_iterations=1000):
    cur_x = x_0
    iteration = 0
    gradient_value = gradient(cur_x)
    norm_of_gradient = np.linalg.norm(gradient_value)
    x_values = [cur_x]
    while norm_of_gradient > precision and iteration < max_iterations:
        h_k = min(eta, gamma * eta / norm_of_gradient)
        cur_x = cur_x - h_k * gradient_value
        iteration = iteration + 1
        x_values.append(cur_x)
        gradient_value = gradient(cur_x)
        norm_of_gradient = np.linalg.norm(gradient_value)

    print('Clipped gradient descent took ', iteration, ' steps')
    return np.array(x_values)
