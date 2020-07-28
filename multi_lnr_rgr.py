__author__ = 'Veerendra'
__date__ = '23-July-2020'
"""
Goal is to predict the house price(dependent variable) using a multi linear regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Lets load the data to solve the problem
int_data = pd.read_csv('C:/Users/veeru/Downloads/Data Scientist/15_Linear Regression-20200718T123140Z-001'
                       '/15 Linear Regression/Assinment/home_data.csv')

# Detect with missing values
missing_values = int_data.isnull()

# Lets count the missing values in each column
for column in missing_values.columns.values.tolist():
    print(column)
    print(missing_values[column].value_counts())  # There is no missing data in the given data set

# Lets check the format of the data types
print(int_data.dtypes)  # All columns data types had the correct data types and we dont need to make any changes

# Lets perform the grouped analysis using binning
# Lets convert the year built column to three categories- Old, Moderate, Recent
bins = bins = np.linspace(min(int_data["yr_built"]), max(int_data["yr_built"]), 4)
group_names = ['Old', 'Moderate', 'Recent']
int_data['yr_built_binned'] = pd.cut(int_data['yr_built'], bins, labels=group_names, include_lowest=True)

# Lets convert the zipcode column to three categories- 98000-50, 98101-150, 98151-200
bins = bins = np.linspace(min(int_data["zipcode"]), max(int_data["zipcode"]), 5)
group_names1 = ['98000-50', '98051-100', '98101-150', '98151-200']
int_data['zipcode_binned'] = pd.cut(int_data['zipcode'], bins, labels=group_names1, include_lowest=True)

# Lets convert the yr_renovated column to two categories - yes or no
int_data['yr_renovated_yes_no'] = np.where(int_data['yr_renovated'] >= 1, 'yes', 'no')

# Lets convert the sqft_basement column to two categories - yes or no
int_data['basement_yes_no'] = np.where(int_data['sqft_basement'] >= 1, 'yes', 'no')

# Regression wont take the categorical data so lets convert the categories that we have created to numerical data
dummy_variable_yr_built = pd.get_dummies(int_data["yr_built_binned"], prefix='yr_built')
dummy_variable_zipcode = pd.get_dummies(int_data["zipcode_binned"], prefix='zipcode')
dummy_variable_yr_renovated = pd.get_dummies(int_data["yr_renovated_yes_no"], prefix='yr_renovated')
dummy_variable_yr_basement = pd.get_dummies(int_data["basement_yes_no"], prefix='basement')


# Lets add the dummy variables to the data
int_data = pd.concat([int_data, dummy_variable_yr_built, dummy_variable_zipcode, dummy_variable_yr_renovated
                         , dummy_variable_yr_basement], axis=1)

# Lets drop the columns that are not useful for further analysis
int_data.drop(['id', 'date', 'yr_built', 'zipcode', 'yr_renovated', 'sqft_living15', 'sqft_lot15',
               'yr_built_binned', 'zipcode_binned', 'yr_renovated_yes_no', 'basement_yes_no'
                  ,'sqft_above', 'sqft_basement'], axis=1, inplace=True)

# print(int_data.corr())
# Lets save the data
int_data.to_csv('C:/Users/veeru/Downloads/Data Scientist/15_Linear Regression-20200718T123140Z-001'
                     '/15 Linear Regression/Assinment/cleaned_data.csv')

# Lets examine the correlation between price and independent variable

# price vs bedrooms
int_data[['price', 'bedrooms']].corr()  # showed positive linear relationship, let's find the scatterplot
sns.regplot(x="bedrooms", y="price", data=int_data)
plt.show()

# price vs bathrooms
int_data[['price', 'bathrooms']].corr()  # showed positive linear relationship, let's find the scatterplot
sns.regplot(x="bathrooms", y="price", data=int_data)
plt.show()

# price vs sqft_living
int_data[['price', 'sqft_living']].corr()  # showed positive linear relationship, let's find the scatterplot
sns.regplot(x="sqft_living", y="price", data=int_data)
plt.show()

# price vs sqft_lot
int_data[['price', 'sqft_lot']].corr()  # showed positive linear relationship, let's find the scatterplot
sns.regplot(x="sqft_lot", y="price", data=int_data)
plt.show()

# price vs floors
int_data[['price', 'floors']].corr()  # showed positive linear relationship, let's find the scatterplot
sns.regplot(x="floors", y="price", data=int_data)
plt.show()

# price vs waterfront
int_data[['price', 'waterfront']].corr()  # showed positive linear relationship, let's find the scatterplot
sns.regplot(x="waterfront", y="price", data=int_data)
plt.show()

# price vs view
int_data[['price', 'view']].corr()  # showed positive linear relationship, let's find the scatterplot
sns.regplot(x="view", y="price", data=int_data)
plt.show()

# price vs grade
int_data[['price', 'grade']].corr()  # showed positive linear relationship, let's find the scatterplot
sns.regplot(x="grade", y="price", data=int_data)
plt.show()

# price vs condition
int_data[['price', 'condition']].corr()  # showed weak linear relationship, let's find the scatterplot
sns.regplot(x="condition", y="price", data=int_data)
plt.show()

# price vs yr_built_old
int_data[['price', 'yr_built_Old']].corr()  # showed weak linear relationship, let's find the scatterplot
sns.regplot(x="yr_built_Old", y="price", data=int_data)
plt.show()

# price vs yr_built_Moderate
int_data[['price', 'yr_built_Moderate']].corr()  # showed negative linear relationship,let's find the scatterplot
sns.regplot(x="yr_built_Moderate", y="price", data=int_data)
plt.show()

# price vs yr_built_Recent
int_data[['price', 'yr_built_Recent']].corr()  # showed positive linear relationship,let's find the scatterplot
sns.regplot(x="yr_built_Recent", y="price", data=int_data)
plt.show()

# price vs zipcode_98000-50
int_data[['price', 'zipcode_98000-50']].corr()  # showed weak relationship,let's find the scatterplot
sns.regplot(x="zipcode_98000-50", y="price", data=int_data)
plt.show()

# price vs zipcode_98051-100
int_data[['price', 'zipcode_98051-100']].corr()  # showed weak relationship,let's find the scatterplot
sns.regplot(x="zipcode_98051-100", y="price", data=int_data)
plt.show()

# price vs zipcode_98051-100
int_data[['price', 'zipcode_98101-150']].corr()  # showed weak relationship,let's find the scatterplot
sns.regplot(x="zipcode_98101-150", y="price", data=int_data)
plt.show()

# price vs zipcode_98151-200
print(int_data[['price', 'zipcode_98151-200']].corr())  # showed negative relationship(-0.08),let's find the scatterplot
sns.regplot(x="zipcode_98151-200", y="price", data=int_data)
plt.show()

# price vs yr_renovated_no
int_data[['price', 'yr_renovated_no']].corr()  # showed negative relationship(-0.12),let's find the scatterplot
sns.regplot(x="yr_renovated_no", y="price", data=int_data)
plt.show()

# price vs yr_renovated_yes
int_data[['price', 'yr_renovated_yes']].corr()  # showed positive relationship(+0.12),let's find the scatterplot
sns.regplot(x="yr_renovated_yes", y="price", data=int_data)
plt.show()

# price vs basement_no
int_data[['price', 'basement_no']].corr()  # showed negative relationship(-0.18),let's find the scatterplot
sns.regplot(x="basement_no", y="price", data=int_data)
plt.show()

# price vs basement_yes
int_data[['price', 'basement_yes']].corr()  # showed positive relationship(+0.18),let's find the scatterplot
sns.regplot(x="basement_yes", y="price", data=int_data)
plt.show()

# price vs lat
int_data[['price', 'lat']].corr()  # showed positive relationship(+0.30),let's find the scatterplot
sns.regplot(x="lat", y="price", data=int_data)
plt.show()

# price vs long
int_data[['price', 'long']].corr()  # showed weak positive relationship(+0.02),let's find the scatterplot
sns.regplot(x="long", y="price", data=int_data)
plt.show()

# Lets calculate the P-value to check the correlation between price & bedrooms variables is statistically significant
# Lets choose a significance level of 0.05, which means that we are 95% confident that the correlation between the
# variables is significant.

# By convention, when the

# p-value is  <  0.001: we say there is strong evidence that the correlation is significant.
# the p-value is  <  0.05: there is moderate evidence that the correlation is significant.
# the p-value is  <  0.1: there is weak evidence that the correlation is significant.
# the p-value is  >  0.1: there is no evidence that the correlation is significant.

from scipy import stats
# price vs bedrooms
pearson_coef, p_value = stats.pearsonr(int_data['bedrooms'], int_data['price'])
#print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) # p_value<0.001,
# correlation between price & bedrooms variables is statistically significant, although the linear relationship isn't
# extremely strong (~0.308)

# price vs bathrooms
pearson_coef, p_value = stats.pearsonr(int_data['bathrooms'], int_data['price']) # p_value<0.001,
# correlation between price & bedrooms variables is statistically significant, although the linear relationship is
# moderately strong (~0.528)

# price vs sqft_living
pearson_coef, p_value = stats.pearsonr(int_data['sqft_living'], int_data['price']) # p_value<0.001,
# correlation between price & bedrooms variables is statistically significant, although the linear relationship is
# strong (~0.702)

# price vs sqft_lot
pearson_coef, p_value = stats.pearsonr(int_data['sqft_lot'], int_data['price'])# p_value<0.001,
# correlation between price & sqft_lot variables is statistically significant, although the linear relationship is
# weak (~0.08)

# price vs floors
pearson_coef, p_value = stats.pearsonr(int_data['floors'], int_data['price']) # p_value<0.001,
# correlation between price & floors variables is statistically significant, although the linear relationship isn't
#  extremely strong(~0.256)

# price vs waterfront
pearson_coef, p_value = stats.pearsonr(int_data['waterfront'], int_data['price']) # p_value<0.001,
# correlation between price & waterfront variables is statistically significant, although the linear relationship isn't
#  extremely strong(~0.266)

# price vs view
pearson_coef, p_value = stats.pearsonr(int_data['view'], int_data['price'])# p_value<0.001,
# correlation between price & waterfront variables is statistically significant, although the linear relationship isn't
#  extremely strong(~0.397)

# price vs grade
pearson_coef, p_value = stats.pearsonr(int_data['grade'], int_data['price']) # p_value<0.001,
# correlation between price & waterfront variables is statistically significant, although the linear relationship is
#  strong(~0.667)

# price vs condition
pearson_coef, p_value = stats.pearsonr(int_data['condition'], int_data['price']) # p_value<0.001,
# correlation between price & waterfront variables is statistically significant, although the linear relationship is
#  weak(~0.036)

# price vs yr_built_old
pearson_coef, p_value = stats.pearsonr(int_data['yr_built_Old'], int_data['price']) # p_value<0.001,
# correlation between price & waterfront variables is statistically significant, although the linear relationship is
#  weak(~0.069)

# price vs yr_built_Moderate
pearson_coef, p_value = stats.pearsonr(int_data['yr_built_Moderate'], int_data['price']) # p_value<0.001,
# correlation between price & yr_built_Moderate variables is statistically significant, although the linear relationship
# isn't extremely strong(~-0.148)

# price vs yr_built_Recent
pearson_coef, p_value = stats.pearsonr(int_data['yr_built_Recent'], int_data['price'])# p_value<0.001,
# correlation between price & yr_built_Recent variables is statistically significant, although the linear relationship is
#  weak(~0.09)

# price vs zipcode_98000-50
pearson_coef, p_value = stats.pearsonr(int_data['zipcode_98000-50'], int_data['price']) # p_value>0.1,
# correlation between price & zipcode_98000-50 variables isn't statistically significant, although the linear
# relationship is weak(~0.01)

# price vs zipcode_98051-100
pearson_coef, p_value = stats.pearsonr(int_data['zipcode_98051-100'], int_data['price']) # p_value>0.1,
# correlation between price & zipcode_98051-100 variables isn't statistically significant, although the linear
# relationship is weak(~0.002)

# price vs zipcode_98101-150
pearson_coef, p_value = stats.pearsonr(int_data['zipcode_98101-150'], int_data['price']) # p_value<0.001,
# correlation between price & yr_built_Recent variables is statistically significant, although the linear relationship
# is weak(~0.04)

# price vs zipcode_98151-200
pearson_coef, p_value = stats.pearsonr(int_data['zipcode_98151-200'], int_data['price']) # p_value<0.001,
# correlation between price & yr_built_Recent variables is statistically significant, although the linear relationship
# is weak(~-0.08)

# price vs yr_renovated_no
pearson_coef, p_value = stats.pearsonr(int_data['yr_renovated_no'], int_data['price']) # p_value<0.001,
# correlation between price & yr_built_Recent variables is statistically significant, although the linear relationship
# isn't extremely strong(~-0.12)

# price vs yr_renovated_yes
pearson_coef, p_value = stats.pearsonr(int_data['yr_renovated_yes'], int_data['price']) # p_value<0.001,
# correlation between price & yr_built_Recent variables is statistically significant, although the linear relationship
# isn't extremely strong(~0.12)

# price vs basement_no
pearson_coef, p_value = stats.pearsonr(int_data['basement_no'], int_data['price']) # p_value<0.001,
# correlation between price & yr_built_Recent variables is statistically significant, although the linear relationship
# isn't extremely strong(~-0.18)

# price vs basement_yes
pearson_coef, p_value = stats.pearsonr(int_data['basement_yes'], int_data['price'])# p_value<0.001,
# correlation between price & yr_built_Recent variables is statistically significant, although the linear relationship
# isn't extremely strong(~0.18)

# price vs lat
pearson_coef, p_value = stats.pearsonr(int_data['lat'], int_data['price']) # p_value<0.001,
# correlation between price & yr_built_Recent variables is statistically significant, although the linear relationship
# isn't extremely strong(~0.30)

# price vs long
pearson_coef, p_value = stats.pearsonr(int_data['long'], int_data['price']) # p_value>0.001,
# correlation between price & yr_built_Recent variables isn't statistically significant, the linear relationship
# isn't extremely strong(~0.02)

# We now have a better idea about the data and what variables are important to predict the house price
# Now lets narrow down the variables by removing the variables showed weak linear relation


final_data = int_data.drop(['sqft_lot', 'condition', 'yr_built_Old',
                            'zipcode_98000-50',
                            'zipcode_98051-100'
                            ,'zipcode_98101-150', 'long'], axis =1)

# Lets save the data
final_data.to_csv('C:/Users/veeru/Downloads/Data Scientist/15_Linear Regression-20200718T123140Z-001'
                        '/15 Linear Regression/Assinment/final_data.csv')


# Lets split the data into train set and test set
X = final_data.drop(['price'], axis = 1)
y = final_data['price']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("number of test samples :", X_test.shape[0])
print("number of training samples:",X_train.shape[0])

# Lets create linear regression object
from sklearn.linear_model import LinearRegression
lre=LinearRegression()

# Lets train the model
lre.fit(X_train, y_train)
print(lre.score(X_train, y_train))
print(lre.score(X_test, y_test))

# r^2 of linear regression model for test and training data showed that ~70% of variation in house price is explained
# by the variation in independent variables

# Lets check the model accuracy

# Lets find out how well the model predicts when compared with actual values
lr = LinearRegression()
lr.fit(X_train, y_train)

# Prediction using training data
yhat_train = lr.predict(X_train)
print(len(yhat_train))

# Prediction using testing data
yhat_test = lr.predict(X_test)
print(yhat_test)

# Lets Plot the predicted values using the training data compared to the training data.
ax1 = sns.distplot(y_train, hist=False, color="r", label='Actual Values (Train)')
ax2 = sns.distplot(yhat_train, hist=False, color="b", label='Predicted Values (Train)')

plt.title('Predicted Value Using Training Data vs Training Data Distribution')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Houses')
plt.show()

# Lets Plot the predicted values using the test data compared to the test data.
ax3 = sns.distplot(y_test, hist=False, color="r", label='Actual Values (Test)')
ax4 = sns.distplot(yhat_test, hist=False, color="b", label='Predicted Values (Test)')

plt.title('Predicted Value Using Testing Data vs Testing Data Distribution')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Houses')
plt.show()

# Comparing the distribution plots of test plot and training plots; its evident both plots showed the actual values
# distribution shape is little different from actual values to predicted values. We have a chance to improve
# the accuracy of the model

