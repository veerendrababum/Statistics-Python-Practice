__author__ = 'Veerendra'
__date__ = '12/July/2020'
'''
Lets explore the Gradient Descent and using the algorithm let's find out the parameters of the equation that reduce the 
cost function
'''
import numpy as np
import matplotlib.pyplot as plt

class find_param_lin_eq():
    def __init__(self, X, y, theta, learning_rate, iterations):
        self.X = X
        self.y = y
        self.theta = theta
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.output = {}
    def gradient_descent(self):
        '''
        Function to get the linear equation parameters that reduces the cost function using gradient descent algorithm
        :param X: Independent variable
        :param y: Target/dependent variable
        :param theta: parameters used for the predictions
        :param learning_rate: Size of the each step
        :param iterations: Iteration of loop to get the local minimum:return: Parameters and final cost function value
        or local minimum
        '''
        X = self.X
        y = self.y
        theta = self.theta
        learning_rate = self.learning_rate
        iterations = self.iterations
        # Lets add the value "1" to X since the all 1st column values of X must be "1" to get multiply with the first
        # row element of theta vector
        x_a = np.c_[np.ones((len(X), 1)), X]
        #print(x_a)

        # Lets find out the length of the target variable
        m = len(y)

        # Create an empty array to add the cost value for each predictions using theta
        cost_history = []

        # Create an empty array to add theta values used for each prediction
        theta_history = [theta]

        # Lets iterate the loop and get the best parameters using Gradient Descent
        for i in range(iterations):
           predictions = x_a.dot(theta)
           error = predictions - y
           cost = np.sum(np.square(error)) * 1 / (2*m)
           cost_history.append(cost)
           theta = theta - (learning_rate * 1 / m) * x_a.T.dot(predictions - y)
           theta_history.append(theta)

        # Lets find out the best cost and theta values
        best_theta = theta_history[-1]
        best_cost = cost_history[-1]

        report = {
            'least_cost': best_cost,
            'theta[1]': best_theta[1],
            'theta[0]': best_theta[0],
        }

        self.output = report
        return report, cost_history


# Creating a row vector of a random sample
X = np.random.rand(100, 1)

# Linear equation of y with parameters
y = 2 * X + 3

# Lets get the random Theta values to use for predictions
theta = np.random.rand(2, 1)

cost_details = find_param_lin_eq(X, y, theta, 0.01, 1000)
report, cost_history = cost_details.gradient_descent()
print(report)

# Lets find the best value for the iterations by plotting the cost_history using the instance of "find_param_lin_eq"
# to get the cost history
def plot_graph(cost_history):
    cost_function = plt.plot(cost_history)
    plt.xlabel('Number Of Iterations')
    plt.ylabel('Cost Values')
    plt.title('Cost Function J')
    plt.show(cost_function)

plot_graph(cost_history)
# Plot showed that change is cost function is constant after 200 iteration so we don't need to iterate the loop for
# 1000 times

