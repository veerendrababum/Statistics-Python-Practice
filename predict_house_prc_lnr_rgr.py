__author__ = 'Veerendra'
__date__ = '18-July-2020'
"""
Goal is to predict the house price for King County Area using simple linear regression
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# Load data
data = pd.read_csv(
    'C:/Users/veeru/Downloads/Data Scientist/15_Linear Regression-20200718T123140Z-001/15 Linear Regression'
    '/Assinment/home_data.csv')
sample_data = data[['sqft_living', 'price']]

# Let's get the independent and dependent variable
house_price = sample_data['price']  # Dependent variable
sqr_feet = sample_data['sqft_living']  # Independent variable

# Lets split the data into test set and training set
from sklearn.model_selection import train_test_split

sqr_feet_train, sqr_feet_test, house_price_train, house_price_test = train_test_split(sqr_feet, house_price,
                                                                                      test_size=0.2, random_state=0)
# Convert the pandas series into a vector to make the calculations simple
sqr_feet_train = np.array([sqr_feet_train])
sqr_feet_train = sqr_feet_train.reshape((-1, 1))

sqr_feet_test = np.array([sqr_feet_test])
sqr_feet_test = sqr_feet_test.reshape((-1, 1))

house_price_train = np.array([house_price_train])
house_price_train = house_price_train.reshape((-1, 1))
house_price_test = np.array([house_price_test])
house_price_test = house_price_test.reshape((-1, 1))
house_price_train = np.divide(house_price_train, 1000)
house_price_test = np.divide(house_price_test, 1000)

# Lets plot the scatter plot to visualize the relation between house price and square feet of the house
graph = plt.scatter(sqr_feet_train, house_price_train)
plt.xlabel('Sqaure Feet')
plt.ylabel('House Price')
plt.title('Scatter plot(House Price vs Sqaure Feet) ')
plt.show(graph)

# Generalised line equation y = mx + c where m is the slope of the equation and  c is the y intercept

# Lets estimate the parameters
sample_size = len(house_price_train)
sum_squares = sqr_feet_train.T.dot(sqr_feet_train)

m = ((sqr_feet_train.T.dot(house_price_train)) - sample_size * np.mean(sqr_feet_train) * np.mean(house_price_train)) / \
    ((sqr_feet_train.T.dot(sqr_feet_train)) - sample_size * (np.square(np.mean(sqr_feet_train))))

c = np.mean(house_price_train) - m * np.mean(sqr_feet_train)

# Lets predict the house price values using square feet test set with estimated equation
house_price_pred = m * sqr_feet_test + c

# Lets find the sum of square errors
SSE = np.sum(np.square(house_price_test - house_price_pred))

# Let check the hypothesis, consider the significance level alpha = 0.05
# Null hypothesis- There is no relation between house price and square feet living
# Alternative Hypothesis- There is a relation between house price and square feet living

standard_error_slope = np.sqrt(SSE/len(house_price_test)-2)
t_val = m / standard_error_slope
pval = scipy.stats.t.sf(np.abs(t_val), len(house_price_test)-2)

# Lets find out the coefficient of determination(R^2)

# First find out the total sum of the squares
SST = np.sum(np.square(house_price_test - np.mean(house_price_test)))

# Lets find out the sum of the squares of regression
SSR = np.sum(np.square(house_price_pred - np.mean(house_price_test)))

# Coefficient of determination R^2 = sum of the squares of regression / total sum of the squares
r_sqr = SSR / SST  # r_sqr = 0.5490404710567703, 0.54% of variation in the house price is explained by fitted regression
# line

# correlation coefficient
r = np.sqrt(r_sqr)  # r value is greater than 0 and there is a positive correlation between the house
# price and square feet living

# Lets find the confidence interval

confidence_interval_low_bound = m + 2.3060 * standard_error_slope
confidence_interval_high_bound = m - 2.3060 * standard_error_slope

report = {
    'Coefficient': m,
    'y-intercept': c,
    'standard_error_slope': standard_error_slope,
    't_val': t_val,
    'p_val': pval,
    'total_sum_of_the_squares': SST,
    'sum_squares_of_regression': SSR,
    'sum of square errors': SSE,
    'r^2': r_sqr,
    'r':r
}
print(report)