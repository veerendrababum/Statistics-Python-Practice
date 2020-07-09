__author__ = 'Veerendra'
__date__ = '08-July-2020'

'''
The goal is to identify and return the peak and valley values of an random sample
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def find_peak_low_points(random_sample):
    """
    Function to find out the peak and valley values
    :param random_sample: Input list to get the peak and low points
    :return: Output lists of peak and valley values in a sample
    """
    # Lets get the list of all peak values
    peak_indexes, _ = find_peaks(random_sample)
    all_peak_values = []
    [all_peak_values.append(random_sample[value]) for value in peak_indexes]

    # Lets get the list of all valley values
    low_indexes, _ = find_peaks(-random_sample)
    all_valley_values = []
    [all_valley_values.append(random_sample[value]) for value in low_indexes]

    # Lets find out the highest peak value
    high_peak_value = np.max(all_peak_values)

    # Lets find out the lowest valley value
    low_valley_value = np.min(all_valley_values)

    # Lets plot the peak and valley points for complete data set of values
    plt.plot(random_sample)
    plt.plot(peak_indexes, all_peak_values, "x", color='red')
    plt.plot(low_indexes, all_valley_values, "o", color='blue')
    plt.xlabel('Size Of Random Sample')
    plt.ylabel('Range Of Random Sample')
    plt.title('Peak And Valley Points Of Sample Data')
    plt.show()

    # Lets return the output list
    report = {
        'original_data': random_sample,
        'all_peak_values': all_peak_values,
        'highest_peak_value': high_peak_value,
        'all_valley_values': all_valley_values,
        'lowest_valley_value': low_valley_value
    }

    return report


sample_data = np.random.choice(10000, 10)
result = find_peak_low_points(sample_data)
print(result)
