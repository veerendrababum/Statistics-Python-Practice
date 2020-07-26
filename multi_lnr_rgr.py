__author__ = 'Veerendra'
__date__ = '23-July-2020'
"""
Goal is to predict the house price(dependent variable) using a multi linear regression
"""


import pandas as pd
import numpy as np

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
print(int_data.dtypes) # All columns data types had the correct data types and we dont need to make any changes


# Lets perform the grouped analysis using binning
# Lets convert the year built column to three categories- Old, Moderate, Recent
bins = bins = np.linspace(min(int_data["yr_built"]), max(int_data["yr_built"]), 4)
group_names = ['Old', 'Moderate' , 'Recent']
int_data['yr_built_binned'] = pd.cut(int_data['yr_built'], bins, labels=group_names, include_lowest=True )


# Lets convert the zipcode column to three categories- 98000-50, 98101-150, 98151-200
bins = bins = np.linspace(min(int_data["zipcode"]), max(int_data["zipcode"]), 5)
group_names1 = ['98000-50', '98051-100', '98101-150', '98151-200']
int_data['zipcode_binned'] = pd.cut(int_data['zipcode'], bins, labels=group_names1, include_lowest=True )

# Lets convert the yr_renovated column to two categories - yes or no
int_data['yr_renovated_yes_no']= np.where(int_data['yr_renovated']>=1, 'yes', 'no')

# Lets convert the sqft_basement column to two categories - yes or no
int_data['basement_yes_no']= np.where(int_data['sqft_basement']>=1, 'yes', 'no')

# Regression wont take the categorical data so lets convert the categories that we have created to numerical data
dummy_variable_yr_built = pd.get_dummies(int_data["yr_built_binned"], prefix = 'yr_built')
dummy_variable_zipcode = pd.get_dummies(int_data["zipcode_binned"], prefix = 'zipcode')
dummy_variable_yr_renovated = pd.get_dummies(int_data["yr_renovated_yes_no"], prefix = 'yr_renovated')
dummy_variable_yr_basement = pd.get_dummies(int_data["yr_renovated_yes_no"], prefix = 'yr_renovated')
dummy_variable_condition = pd.get_dummies(int_data["basement_yes_no"], prefix = 'basement')

# Lets add the dummy variables to the data
int_data = pd.concat([int_data, dummy_variable_yr_built, dummy_variable_zipcode, dummy_variable_yr_renovated,
                      dummy_variable_condition,dummy_variable_yr_basement], axis =1)

# Lets drop the columns that are not useful for further analysis
int_data.drop(['id', 'date', 'zipcode', 'yr_built', 'condition', 'yr_renovated', 'sqft_living15', 'sqft_lot15',
               'yr_built_binned', 'zipcode_binned', 'yr_renovated_yes_no', 'basement_yes_no', 'lat',
               'long','sqft_above', 'sqft_basement'], axis = 1, inplace = True)

# Lets save the data
int_data.to_csv('C:/Users/veeru/Downloads/Data Scientist/15_Linear Regression-20200718T123140Z-001'
                       '/15 Linear Regression/Assinment/cleaned_data.csv')