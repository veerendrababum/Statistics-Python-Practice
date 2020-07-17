__author__ = 'Veerendra'
__Date__ = '16-July-2020'

"""
Lets explore the power of central limit theorem- No matter what the population distribution is, the sample means of 
sample distribution will reach out to the normal distribution when the selected samples size is sufficient(n>30)
"""

import numpy as np
import matplotlib.pyplot as plt


def central_limit_theorem(sample_size, iterations):
    # Lets takes the original distribution
    original_dist = np.random.choice(10000, 1000)

    # Lets find out the mean of original distribution
    original_dist_mean = np.mean(original_dist)

    # Lets find the means of random samples taken from original distribution of data
    random_sample_means = []
    for i in range(iterations):
        random_sample = np.random.choice(original_dist, sample_size)
        random_sample_means.append(np.mean(random_sample))

    # Lets find out the mean of random samples mean
    mean_random_sample_means = np.mean(random_sample_means)

    # Lets plot the original distribution
    original_dist_hist = plt.hist(original_dist)
    plt.xlabel('Population Data')
    plt.ylabel('Frequency')
    plt.title('Population Distribution Histogram')
    plt.show(original_dist_hist)

    # Lets plot the sample means distribution
    random_sample_hist = plt.hist(random_sample_means)
    plt.xlabel('Random Sample Means')
    plt.ylabel('Frequency')
    plt.title('Random Sample Means Distribution Histogram')
    plt.show(random_sample_hist)

    report = {
        'population_mean': original_dist_mean,
        'mean_of_random_sample_means': mean_random_sample_means
    }
    return report


output = central_limit_theorem(60, 1000)
print(output)
