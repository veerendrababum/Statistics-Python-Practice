import numpy as np
from scipy import stats

data_set = [26, 33, 65, 28, 34, 55, 25, 44, 50, 36, 26, 37, 43, 62, 35, 38, 45, 32, 28, 34]
#The goal is to find the ouliers of the data set
#Underlying normal distribution, 95% of the data falls within the 2 standard deviation aways from the mean
#z-scores tells us how many standard deviation a data point is away from the mean
z = stats.zscore(data_set)
#Outliers will be the values below the 2 standard deviation from the mean and above the 2 standard deviation above the mean
for i in range(len(z)):
    if( z[i] > 2 or z[i] < -2):
        print('outlier-', data_set[i], 'z-score-', z[i])