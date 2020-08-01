__author__ = 'Veerendra'
__date__ = '07/29/2020'
"""
Goal is to identify the payment transaction is fraud or not and also the probability of payment transaction belongs to
fraud class or not.
We are taking a data set from Kaggle and it has credit card transactions made in two days by european card
holders where we have 492 frauds out of 284,807 transactions. 
It contains 31 features in total. Because of the confidentiality issues 28 features(V1, V2, V3, V4...V28) are 
transformed with PCA. The only features which have not been transformed with PCA are 'Time', 'Amount', 'Class'(Response
variable).
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# Lets load the data
initial_data = pd.read_csv('C:/Users/veeru/Downloads/Data Scientist/My_1st_Payments_Data_Science_Project/'
                           'creditcard.csv')
print(initial_data.head())

# Lets do data pre-processing

# Detect the missing values
missing_values = initial_data.isnull()

# Lets count the missing values in each column
for column in missing_values.columns.values.tolist():
    print(column)
    print(missing_values[column].value_counts())  # There is no missing data in the given data set

# Lets check the format of the data types
print(initial_data.dtypes)  # All columns data types had the correct data types and it makes sense to have data types
# as only int and float since the data set contains only numerical variables

# Lets understand the data distribution of non-transformed data- Time, Amount, and Class

# Plotted the distribution of amount
dist_amount = sns.distplot(initial_data['Amount'], color="r")
plt.title('Distribution of Transaction Amount')
plt.xlabel('Amount')
plt.show() # Distribution shows that most of the transaction are below 70

# Plotted the distribution of time
dist_time = sns.distplot(initial_data['Time'], color="g")
plt.title('Distribution of Transaction Time')
plt.xlabel('Time')
plt.show() # Distribution of time for two days shows that most of the transactions are taken place during the daytime
# and peaked after business hours timing and after that it got slowed down until the next day

# Plotted the distribution of Fraud class
dist_class = sns.countplot(initial_data['Class'], color="b")
plt.title('Distribution of Fraud class')
plt.xlabel('Class(0-Fraud or 1-Not Fraud)')
plt.show() # By seeing the distribution of Class(0-Fraud or 1-Not Fraud) its clear that the data is highly imbalanced.
# Lets figure out, how we can make our next steps to catch the fraudsters

# Exploratory Data analysis
# Lets examine the correlation between the variables to make there is no strong collinearity between the independent
# variables using heat map
plt.figure(figsize=(16,16))
sns.heatmap(initial_data.corr(), annot = False, cmap = 'coolwarm')
plt.show() # From the heat map we have observed that all diagonal elements shown strong collinearity since it shows the
# collinearity between the same variables. We also noticed that there correlation between Time and V3, Amount and V2,
# Amount and V5, Amount and V7, Amount and V20. Lets find out the exact correlation coefficient to make sure whether
# we can consider these correlations or not

initial_data[['Time', 'V3']].corr() # showed negative relationship(~-0.41)
initial_data[['Amount', 'V2']].corr() # showed negative relationship(~-0.53)
initial_data[['Amount', 'V5']].corr() # showed negative relationship(~-0.38)
initial_data[['Amount', 'V7']].corr() # showed negative relationship(~0.39)
initial_data[['Amount', 'V20']].corr() # showed negative relationship(~0.33)
# None of them showed strong correlation that might effect the model results. We can move to the next steps.
# Lets define X, and y for our dataset
X = initial_data.drop(['Class'], axis = 1)
y = initial_data['Class']

# Lets normalize the data. Its important to bring the data to be normally distributed for classification problems and
# distributed normally with mean 0 and standard deviation 1. This helps us to deal with the different ranges of time
# and amount
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
print("number of test samples :", X_test.shape, y_test.shape)
print("number of training samples:",X_train.shape, y_train.shape)

# Lets build our model using LogisticRegression from Scikit-learn package.
from sklearn.linear_model import LogisticRegression
Log_Rgr = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

# Lets predict the response variable values using X_test
yhat = Log_Rgr.predict(X_test)

# Logistic regression gives us the probability of our prediction values. Lets use predict_proba function which gives us
# the first column of probability of Class belongs to "0"(P(Y=0|X)) and the second column of the probability of Class
# belongs to "1"(P(Y=1|X))
yhat_prob = Log_Rgr.predict_proba(X_test)

# Printing the confusion matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
from sklearn import metrics
print(metrics.confusion_matrix(y_test, yhat, labels=[1,0]))
# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_test, yhat, labels=[1,0]))

# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, yhat)

# roc curve and auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# generate a no skill prediction
ns_probs = [0 for _ in range(len(y_test))]

# keep probabilities for the positive outcome only
lr_probs = Log_Rgr.predict_proba(X_test)[:,1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, thresholds = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, thresholds = roc_curve(y_test, lr_probs)

# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# show the legend
plt.legend()

# show the plot
plt.show()











