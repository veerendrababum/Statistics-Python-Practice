__author__ = "Veerendra"
__date__ = "08-Aug-2020"
"""
Problem Statement: The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by 
Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is 
classified as good or bad credit risks according to the set of attributes. 

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
# categorical data. There is a unnamed column which represents the serial number of the rows but this data is
# not required for us since pandas data frame has default indexer. Lets drop the unnamed column from our
# data frame.

# Rename a unnamed column to unknown
intl_data.rename({'Unnamed: 0': 'unknown'}, axis=1, inplace=True)
intl_data.drop(['unknown'], axis=1, inplace=True)
print(intl_data.head())

# And also we have noticed few missing values in the first five rows itself. Lets perform data wrangling to
# get the clean data which can be used to further exploratory data analysis.

# Detect with missing values
missing_values = intl_data.isnull()

# Lets count the missing values in each column
for column in missing_values.columns.values.tolist():
    print(column)
    print(missing_values[column].value_counts())

# Whole columns should be dropped only if most entries in the column are empty. In our dataset, none of the columns
# are empty enough to drop entirely. Out of 10 columns, we have missing data only in two columns (Saving accounts
# - 183) and Checking account - 394).

# There will be chances that customers will have saving account but not checking account and vice versa and also
# some customers might take the loan/credit without having the account. For now, lets fill the NaN values with
# 'none'
for col in ['Saving accounts', 'Checking account']:
    intl_data[col].fillna('none', inplace=True)
print(intl_data.head())

# Lets check the format of the data types
print(intl_data.dtypes)  # All columns data types had the correct data types and we dont need to make any changes

print(intl_data['Age'].value_counts())
print('Unique values in age =', intl_data['Age'].nunique())  # In our data set age is a real valued variable
# ranging from 19 to 75 with 53 unique values.

# What if we only care  about the credit risk between persons young age, middle age, and old (3 types)?
# Can we rearrange them into three ‘bins' to simplify analysis?
bins = np.linspace(min(intl_data["Age"]), max(intl_data['Age']), 4)

# Set group names
group_names = ['Young age', 'Middle age', 'Old age']

# Lets apply the function "cut" the determine what each value of "intl_data['Age']" belongs to.
intl_data['Age-binned'] = pd.cut(intl_data['Age'], bins, labels=group_names, include_lowest=True)
print(intl_data.head(20))

# Lets convert the Job numerical data into categorical data(numeric: 0 - unskilled and non-resident,
# 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)

intl_data['Job'] = intl_data['Job'].replace({
    0: 'unskilled and non-resident',
    1: 'unskilled and resident',
    2: 'skilled',
    3: 'highly skilled'
})

print(intl_data.head())

# Lets analyse the each feature patterns using visualization

# Lets start with the distribution of risk across the age-binned
sns.countplot(x="Age-binned", hue="Risk", data=intl_data)
plt.show()

# Lets see the distribution of Risk across the sex
sns.countplot(x="Sex", hue="Risk", data=intl_data)
plt.show() # We have observed that risk of credit is high for female when compared with male

# Lets see the distribution of Risk across the housing
sns.countplot(x="Housing", hue="Risk", data=intl_data)
plt.show() # We have observed that there is high risk associated with rent and free housing customers than the own
# housing customers

# Lets see the distribution of Risk across the Job(Job (numeric: 0 - unskilled and non-resident,
# 1 - unskilled and resident, 2 - skilled, 3 - highly skilled))
sns.countplot(x="Job", hue="Risk", data=intl_data)
plt.show() # Distribution showed that skilled job type customers are the highest good credit holders and good credit
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

sns.boxplot(x="Checking account", y="Credit amount", data=intl_data, hue="Risk")
plt.show() # Box plot showed that the highest bad credit holders are in moderate, little , none(keep a note that
# we have converted NaN to none value) and also the higher the credit amount the more likely to to be a bad credit
# for moderate, little , none. However, rich category showed less bad credits.

sns.boxplot(x="Saving accounts", y="Credit amount", data=intl_data, hue="Risk")
plt.show() #

sns.boxplot(x="Duration", y="Credit amount", data=intl_data, hue="Risk")
plt.show() # Box plot clearly shows that as long as the duration increases the credit amount is increased and also
# simultaneously bad credits increased.

# With help of EDA, we understand the importance of each feature and also to improve the accuracy of our prediction
# its important to create the new boolean features(0 or 1) for each categorical variable and this also helps
# to train the model without any format errors since we need to train the machine learning using numerical data

# Lets do feature engineering

# Lets create the dummy values for categorical variable

# Lets use the panda's method 'get_dummies' to assign numerical values to different categories.

dummies_columns = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Age-binned']
for col in dummies_columns:
    intl_data = intl_data.merge(pd.get_dummies(intl_data[col], drop_first=False, prefix=str(col)),
                                left_index=True, right_index=True)

# Lets remove the redundant categorical variables
intl_data.drop(dummies_columns, axis=1, inplace=True)
intl_data.drop('Age', axis=1, inplace=True)
print(intl_data.info())

# Lets convert the target variable(Risk) columns values good to "1" and bad to "0"

intl_data['Risk'] = intl_data['Risk'].replace({
    'good': 0,
    'bad': 1
})
print(intl_data['Risk'].head())

# Lets examine the correlation between the variables to make sure there is no strong collinearity between the
# independent variables using heat map
plt.figure(figsize=(16, 16))
sns.heatmap(intl_data.corr(), annot=False, cmap='coolwarm')
plt.show()  # From the heat map we have observed that all diagonal elements shown strong collinearity since it shows
# the collinearity between the same variables. We also noticed that there correlation between Duration and Credit
# amount, Age-binned_Young age  and Middle age. Lets find out the exact correlation coefficient to make sure whether
# we can consider these correlations or not.

print(intl_data[['Duration', 'Credit amount']].corr())  # It showed positive correlation of 0.62 but it might not
# effect the model results.
print(intl_data[['Age-binned_Young age', 'Age-binned_Middle age']].corr())  # It showed negative correlation of
# -0.85. Lets keep this value for now and based one the model accuracy, we will decide what to do next.
print(intl_data[['Sex_male', 'Sex_female']].corr())  # Its showed very strong negative correlation of -1. Lets
# drop one column - Sex_female

mdl_data = intl_data.drop('Sex_female', axis=1)
print(mdl_data.info())

# Before starting the model training. Lets perform one last step, plotting the distribution of target variable to
# check whether the data is balanced or not.
# sns.countplot(x="Age-binned", hue="Risk", data=intl_data)
# plt.show()
dist_class = sns.countplot(mdl_data['Risk'], color="b")
plt.title('Distribution of Credit Risk class')
plt.xlabel('Class(1-Good Credit or 0-Bad Credit)')
plt.show()  # Data is decently balanced with 700 good credit and 300 bad credits out of 1000 credits

# Lets use different classification machine algorithms to check and make sure which model performs better on
# our dataset

# Lets first start with Logistic Regression

# Lets define X, and y for our dataset
X = mdl_data.drop(['Risk'], axis=1)
y = mdl_data['Risk']

# Lets normalize the data. Its important to bring the data to be normally distributed for classification problems
# and distributed normally with mean 0 and standard deviation 1. This helps us to deal with the different ranges
# of variables(Ex:Credit Amount)
from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("number of test samples :", X_test.shape, y_test.shape)
print("number of training samples:", X_train.shape, y_train.shape)

# Lets build our Logistic Regression model using training data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Lets find out the best parameters using GridSearchCV
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_log_rgr = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=10000),
                            log_reg_params, cv=5, scoring='roc_auc')
grid_log_rgr.fit(X_train, y_train)

# Lets predict the response variable values using undersampled test data
yhat_trn = grid_log_rgr.predict(X_train)

# Logistic regression gives us the probability of our prediction values. Lets use predict_proba function which gives us
# the first column of probability of Class belongs to "0"(P(Y=0|X)) and the second column of the probability of Class
# belongs to "1"(P(Y=1|X))
yhat_prob_trn = grid_log_rgr.predict_proba(X_train)

# Logistic Regression best estimator
print(grid_log_rgr.best_estimator_)

# Lets print the model accuracy
print(grid_log_rgr.best_score_)

# Calculate the confusion matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
from sklearn import metrics

conf_mtrx_undr_smpl = metrics.confusion_matrix(y_train, yhat_trn, labels=[1, 0])

# Lets plot the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Created reusable code and it will be useful in the next steps
def plt_cnfs_mtx(confusion_matrix, title):
    ax = plt.subplot()
    sns.heatmap(confusion_matrix, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues);  # annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title(title);
    ax.xaxis.set_ticklabels(['Good Credit', 'Bad Credit']);
    ax.yaxis.set_ticklabels(['Good Credit', 'Good Credit']);
    plt.show()


plt_cnfs_mtx(conf_mtrx_undr_smpl, 'Confusion Matrix Logistic Regression Trained Data')

# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_train, yhat_trn, labels=[1, 0]))

# Logistic regression model showed recall percentage of 59 to detect the bad creditors and with overall accuracy
# of the model as 74 on train data

# Lets see, how the model will act with test data
yhat_log_rgr_test = grid_log_rgr.predict(X_test)
conf_mtrx_log_rgr_test = metrics.confusion_matrix(y_test, yhat_log_rgr_test, labels=[1, 0])
plt_cnfs_mtx(conf_mtrx_log_rgr_test, 'Confusion Matrix Logistic Regression Test Data')
print(metrics.classification_report(y_test, yhat_log_rgr_test, labels=[1, 0]))

# On test data, Logistic regression model showed overall accuracy of 71% and recall percentage(51%) even went low
# when compared with the trained data recall percentage of 59

# Lets use K Nearest Neighbour algorithm to classify the response variable

# Lets train the KNN model
from sklearn.neighbors import KNeighborsClassifier

# Use GridsearchCV to get the best parameters
from sklearn.model_selection import GridSearchCV

knears_params = {"n_neighbors": list(range(2, 10, 1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

# Lets train the model using train data
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)

# KNears best estimator
knears_neighbors = grid_knears.best_estimator_
print(grid_knears.best_estimator_)

# Lets print the model accuracy
print(grid_knears.best_score_)

# Lets predict the values
yhat_knn_trn = grid_knears.predict(X_train)

# Printing the confusion matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
from sklearn import metrics

conf_mtrx_knn = metrics.confusion_matrix(y_train, yhat_knn_trn, labels=[1, 0])

# Lets plot the confusion matrix of KNN trained data
plt_cnfs_mtx(conf_mtrx_knn, 'Confusion Matrix KNN Trained Data')
print(metrics.classification_report(y_train, yhat_knn_trn, labels=[1, 0]))

# Results of KNN on trained data aren't good when compared with the Logistic Regression with the low recall
# percentage of 39 to detect the bad creditors

# Lets test the KNN model
yhat_knn_test = grid_knears.predict(X_test)

# Lets plot the confusion matrix of KNN test data
conf_mtrx_knn_test = metrics.confusion_matrix(y_test, yhat_knn_test, labels=[1, 0])
plt_cnfs_mtx(conf_mtrx_knn_test, 'Confusion Matrix KNN Test Data')
print(metrics.classification_report(y_test, yhat_knn_test, labels=[1, 0]))

# The KNN model even performed worse on test data with recall percentage of 27


# Lets build the decision tree model
from sklearn.tree import DecisionTreeClassifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)),
               "min_samples_leaf": list(range(5, 7, 1))}
grid_dec_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
# Lets train the model using train data
grid_dec_tree.fit(X_train, y_train)

# Decision tree best estimator
print(grid_dec_tree.best_estimator_)

# Lets print the model accuracy
print(grid_dec_tree.best_score_)

# Lets predict the outcome of the model using train data
yhat_grid_dcc_trn = grid_dec_tree.predict(X_train)

# Lets find the probability of each outcome using predict_proba
yhat_prb_grid_dcc_trn = grid_dec_tree.predict_proba(X_train)

# Lets plot the confusion matrix
conf_mtrx_grid_dcc_trn = metrics.confusion_matrix(y_train, yhat_grid_dcc_trn, labels=[1, 0])
plt_cnfs_mtx(conf_mtrx_grid_dcc_trn, 'Confusion Matrix Decision Tree Train Data')

# Lets find out the matrices of the trained model
print(metrics.classification_report(y_train, yhat_grid_dcc_trn, labels=[1, 0]))

# On trained data, decision tree model performed better than the KNN with overall accuracy of 71 and recall score of
# 46.

# Lets predict the outcome of the model using the test data
yhat_grid_dcc_test = grid_dec_tree.predict(X_test)

# Lets find the probability of each outcome using predict_proba function
yhat_grid_prb_dcc_test = grid_dec_tree.predict_proba(X_test)

# Lets plot the confusion matrix
conf_mtrx_grid_dcc_test = metrics.confusion_matrix(y_test, yhat_grid_dcc_test, labels=[1, 0])
plt_cnfs_mtx(conf_mtrx_grid_dcc_test, 'Confusion Matrix Decision Tree Test Data')

# Lets find out the matrices of the trained model
print(metrics.classification_report(y_test, yhat_grid_dcc_test, labels=[1, 0]))

# On test data, decision tree model showed same accuracy as trained data with recall score of 46.

# So far Logistic regression showed best results than the KNN and Decision

# Next step: Train and test the performance of SVM and Naive-Bayes on our dataset
