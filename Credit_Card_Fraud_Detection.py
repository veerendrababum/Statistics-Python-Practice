__author__ = 'Veerendra'
__date__ = '07/29/2020'
"""
Goal is to build a machine learning model that can label the payment transaction is fraud or not

We are taking a data set from Kaggle and it has credit card transactions made in two days by european card
holders where we have 492 frauds out of 284,807 transactions. 

It contains 31 features in total. Because of the confidentiality issues 28 features(V1, V2, V3, V4...V28) are 
transformed with PCA. The only features which have not been transformed with PCA are 'Time', 'Amount', 'Class'(Response
variable- Fraud - 1, Not Fraud - 0).

"""
import pandas as pd
from matplotlib import pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import numpy as np

# Lets load the data
# Link to download the data - https://www.kaggle.com/mlg-ulb/creditcardfraud
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
plt.xlabel('Class(1-Fraud or 0-Not Fraud)')
plt.show() # By seeing the distribution of Class(1-Fraud or 0- Not Fraud) its clear that the data is highly imbalanced.
# Lets figure out, how we can make our next steps to catch the fraudsters

# Exploratory Data analysis
# Lets examine the correlation between the variables to make sure there is no strong collinearity between the
# independent variables using heat map
plt.figure(figsize=(16,16))
sns.heatmap(initial_data.corr(), annot=False,cmap = 'coolwarm')
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
print("number of test samples :", X_test.shape, y_test.shape)
print("number of training samples:",X_train.shape, y_train.shape)

# As we observed, this data set is imbalanced so I assume undersampling will helps us to predict more accurate results.
# However, there might be chance to miss the important information when we do undersampling. Lets measure the accuracy
# by building a separate logistic regression model by training with skewed data(80% data from whole data) and another
# logistic regression model trained with undersampled data taken from the training set (80% data from whole data)
# to predict the fraud transaction using undersampled test data and test data(remaining 20% of data)

# Lets first start with undersampled data
X_train_undersample, y_train_undersample = RandomUnderSampler(random_state=0).fit_sample(X_train,y_train)
X_test_undersample, y_test_undersample = RandomUnderSampler(random_state=0).fit_sample(X_test,y_test)

# Lets build our model using undersampled training data and test data
from sklearn.linear_model import LogisticRegression
Log_Rgr_Undr_Smpl = LogisticRegression(C=0.01, solver='liblinear').fit(X_train_undersample, y_train_undersample)

# Lets predict the response variable values using undersampled test data
yhat_undersample = Log_Rgr_Undr_Smpl.predict(X_test_undersample)

# Logistic regression gives us the probability of our prediction values. Lets use predict_proba function which gives us
# the first column of probability of Class belongs to "0"(P(Y=0|X)) and the second column of the probability of Class
# belongs to "1"(P(Y=1|X))
yhat_prob_undersample = Log_Rgr_Undr_Smpl.predict_proba(X_test_undersample)

# Calculate the confusion matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
from sklearn import metrics
conf_mtrx_undr_smpl = metrics.confusion_matrix(y_test_undersample, yhat_undersample, labels=[1,0])

# Lets plot the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Created reusable code and it will be useful in the next steps
def plt_cnfs_mtx(confusion_matrix):
    ax= plt.subplot()
    sns.heatmap(confusion_matrix, annot=True, fmt= 'd',ax = ax, cmap = plt.cm.Blues); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['Fraud', 'Legitimate']); ax.yaxis.set_ticklabels(['Fraud', 'Legitimate']);
    plt.show()

plt_cnfs_mtx(conf_mtrx_undr_smpl)

# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_test_undersample, yhat_undersample, labels=[1,0]))

# True positive rate or recall percentage is the important measure to consider since we are building the model to find
# out the fraud transactions. Using undersample test data taken from X-test data(20% data divided from original data).
# Model showed good recall percentage of 91.

# Lets see how the model trained using undersampled data will predict the fraud transaction on total
# X_test data((20% data divided from original data))

# Lets predict the response variable values using X_test
yhat = Log_Rgr_Undr_Smpl.predict(X_test)

# Logistic regression gives us the probability of our prediction values. Lets use predict_proba function which gives us
# the first column of probability of Class belongs to "0"(P(Y=0|X)) and the second column of the probability of Class
# belongs to "1"(P(Y=1|X))
yhat_prob = Log_Rgr_Undr_Smpl.predict_proba(X_test)

# Printing the confusion matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
from sklearn import metrics

conf_mtrx_overall_test_data = metrics.confusion_matrix(y_test, yhat, labels=[1,0])
print(conf_mtrx_overall_test_data)

# Let's plot the confusion matrix
plt_cnfs_mtx(conf_mtrx_overall_test_data)

# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_test, yhat, labels=[1,0]))

# Our trained model using the undersampled data showed better performance on total X_test data
# (20% data divided from original data) with recall percentage of 91. Precision rate went low but model showed
# better performance to find out the fraud transactions and this kind of model might results is asking some extra
# questions to customers(Ex: These kind of situations might trigger while travelling customer made transactions in one
# city and in the next hour they tried to make transaction in another city). Eventhough, the model is giving more
# accurate results to find out the fraud transaction but we need to make our model stable with precision score and
# this can be achieved by finding out the better threshold which is used to classify the transaction. We will
# focus on the next steps.

# Our logistic regression model did good job when trained with undersamled data. Lets see how the model will work when
# we trained with 80% trained data taken from whole data)
from sklearn.linear_model import LogisticRegression
Log_Rgr_Whl_Data = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

# Lets predict the response variable values using X_test
yhat_Whl_Data = Log_Rgr_Whl_Data.predict(X_test)

# Logistic regression gives us the probability of our prediction values. Lets use predict_proba function which gives us
# the first column of probability of Class belongs to "0"(P(Y=0|X)) and the second column of the probability of Class
# belongs to "1"(P(Y=1|X))
yhat_prob_Whl_Data = Log_Rgr_Whl_Data.predict_proba(X_test)
#print(yhat_prob)

# Printing the confusion matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
from sklearn import metrics
conf_mtrx_cmplt_dataset = metrics.confusion_matrix(y_test, yhat_Whl_Data, labels=[1,0])

# Lets plot the confusion matrix
plt_cnfs_mtx(conf_mtrx_cmplt_dataset)

# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_test, yhat_Whl_Data, labels=[1,0]))

# Its clear that the logistic regression model showed low performance when trained with 80% data taken from
# whole data with true fraud detection rate(recall rate) of 57. So model showed accurate results when we trained with
# undersampled data and proved the same when tested with different samples of test data

# Lets see the overall accuracy of the model using ROC curve and AUC
# roc curve and auc
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, recall_score, accuracy_score, precision_score, \
    plot_confusion_matrix

# generate a no skill prediction
ns_probs = [0 for _ in range(len(y_test))]

# keep probabilities for the positive outcome only
lr_probs = Log_Rgr_Undr_Smpl.predict_proba(X_test)[:,1]

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

# Our model showed the great accuracy using ROC curve with AUC score of 0.98. It's the time to improve the model by
# finding out the best threshold and later we will make our model to identify the fraud transactions using new threshold

def model_prfm_diff_thresholds():
    for threshold in np.linspace (0, 1, 20):
        y_predict = Log_Rgr_Undr_Smpl.predict_proba(X_test)[:,1] >= threshold
        f1 = f1_score(y_test,y_predict)
        acc = accuracy_score(y_test,y_predict)
        rec = recall_score(y_test,y_predict)
        pre = precision_score(y_test,y_predict)
        print(threshold, f1, acc, rec, pre)

print(model_prfm_diff_thresholds())

# Looking at the thresholds, we can make our model stabilised if we use thresholds values
# as 0.78(Recall - 86.73, Precision-42.71) or 0.842(Recall - 83.67, Precision-51.25). It will be always good to have
# balanced model even though changing the threshold from 0.5(default) to 0.78 or 0.842 brings down the Recall percentage
# from 91 to 86.73 or 83.67 but the precision score is jumped from 0.05 at threshold- 0.5 to decent percentage at
# threshold- 0.78(Precision-42.71) or threshold- 0.842(Precision-51.25)

# Next steps: We will apply the dataset with some other classification models and then we will accuracy between models.










