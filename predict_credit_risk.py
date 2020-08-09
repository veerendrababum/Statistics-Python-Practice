__author__ = "Veerendra"
__date__ = "08-Aug-2020"
"""
Problem Statement: The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by 
Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is classified 
as good or bad credit risks according to the set of attributes. 

Goal: Predict the new customer belongs to good pr bad credit risk based on the features of the customers. 

Solution: Using machine learning classification technique we can build a best suitable model to classify the future
customer belongs to good or bad credit risks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

# Lets load the data
intl_data = pd.read_csv('C:/Users/veeru/Downloads/Data Scientist/Predict_Credit_Risk'
                        '/datasets_9109_12699_german_credit_data.csv')
print(intl_data.head())

# Now we have loaded the data into data frame. By looking at the columns, we have the mixture of numeric and
# categorical data and unnamed column represents the serial number of the rows but this data is not required for us
# since pandas data frame has default indexer. Lets drop the unnamed column from our data frame.

# Rename a unnamed column to unknown
intl_data.rename({'Unnamed: 0': 'unknown'}, axis=1, inplace=True)
intl_data.drop(['unknown'], axis=1, inplace=True)
print(intl_data.head())

# And also we have noticed few missing values in the first five rows itself. Lets perform data wrangling to
# get the clean data which can be used to further data exploratory analysis.

# Detect with missing values
missing_values = intl_data.isnull()

# Lets count the missing values in each column
for column in missing_values.columns.values.tolist():
    print(column)
    print(missing_values[column].value_counts())

# Whole columns should be dropped only if most entries in the column are empty. In our dataset, none of the columns
# are empty enough to drop entirely. Out of 10 columns, we have missing data only in two columns (Saving accounts
# - 183) and Checking account - 394).

# There will be chances that customers will have saving account but not checking account and vice versa and also some
# customers might take the loan/credit without having the account. For now, lets fill the NaN values with 'none'
for col in ['Saving accounts', 'Checking account']:
    intl_data[col].fillna('none', inplace=True)
print(intl_data.head())

# Lets check the format of the data types
print(intl_data.dtypes)  # All columns data types had the correct data types and we dont need to make any changes

#%matplotlib inline
#import matplotlib as plt
#from matplotlib import pyplot as plt
#veeru = plt.bar(['none', 'little', 'moderate', 'rich'], intl_data['Checking account'].value_counts())

# set x/y labels and plot title
#plt.xlabel("horsepower")
#plt.ylabel("count")
#plt.title("horsepower bins")
#lt.show()
print(intl_data['Age'].value_counts())
print('Unique values in age =', intl_data['Age'].nunique())
# In our data set age is a real valued variable ranging from 19 to 75 with. What if we only care about the credit
# risk between persons young age, middle age, and old (3 types)? Can we rearrange them into three ‘bins' to
# simplify analysis?
bins = np.linspace(min(intl_data["Age"]), max(intl_data['Age']), 4)

# Set group names
group_names = ['Young age', 'Middle age', 'Old age']

# Lets apply the function "cut" the determine what each value of "intl_data['Age']" belongs to.
intl_data['Age-binned'] = pd.cut(intl_data['Age'], bins, labels=group_names, include_lowest=True)
print(intl_data.head(20))


# Lets analyse the each feature patterns using visualization

# Lets start with the distribution of risk across the age-binned
sns.countplot(x="Age-binned", hue="Risk", data=intl_data)
#plt.show()

# Lets see the distribution of Risk across the sex
sns.countplot(x="Sex", hue="Risk", data=intl_data)
#plt.show() # We have observed that risk of credit is high for female when compared with male

# Lets see the distribution of Risk across the housing
sns.countplot(x="Housing", hue="Risk", data=intl_data)
#plt.show() # We have observed that there is high risk associated with rent and free housing customers than the own
# housing customers

# Lets see the distribution of Risk across the Job(Job (numeric: 0 - unskilled and non-resident,
# 1 - unskilled and resident, 2 - skilled, 3 - highly skilled))
sns.countplot(x="Job", hue="Risk", data=intl_data)
#plt.show() # Distribution showed that skilled job type customers are the highest good credit holders and good credit
# holders are less in unskilled groups and there is no surprise to see this result. Skilled workers will be paid
# good and there job is more secured when compared with skilled labour

# Let see the distribution of Risk across Saving account
sns.countplot(x="Saving accounts", hue="Risk", data=intl_data)
plt.show() # Distribution showed that "little" Savings accounts customers are the highest good credit holders.
# However, the same group showed the highest "bad" credit holders. More likely the customers of rich and
# quite rich are classified as  good credit holders and it make sense since the more you have the saving the amount
# there is high possibility to get the lender amount back.

# Let see the distribution of Risk across Checking account
sns.countplot(x="Checking account", hue="Risk", data=intl_data)
plt.show()

# Lets see how much credit amount given across different customer features are turned to be good or bad credit
sns.boxplot(x="Sex", y="Credit amount", data=intl_data, hue="Risk")
plt.show() # Box plot showed that lower the credit amount turned to be good credit for both male and female. The
# credit amount taken 12500 turned to be a bad credit for male.

sns.boxplot(x="Purpose", y="Credit amount", data=intl_data, hue="Risk")
plt.show() # Box plot showed that the highest amount credited for car, business and vacations are turned into
# bad credits.

sns.boxplot(x="Job", y="Credit amount", data=intl_data, hue="Risk")
plt.show() # Box plot showed that the unskilled people paid less loan amount when compared with skilled people.
# Its obvious that loan creditors trust the skilled people since they have more job security and high paid when
# compared with unskilled labour. However, skilled people took high loan amounts are turned to be bad credits.

