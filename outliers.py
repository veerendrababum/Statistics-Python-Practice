__author__ = 'Veerendra'
__date__ = '05-July-2020'

'''
Created functions for different methods to identify the outliers
'''
import scipy.stats
import matplotlib.pyplot as plt


def remove_outliers_z_score(random_sample):
    """
    A function to identify and remove the outliers from a random_sample using z_score
    :param random_sample: input list to identify and remove outliers to improve skewness
    :return: Output list after removing the outliers and skewness
    """

    # Calculated the z_scores for the list of elements in random_sample
    z_scores = scipy.stats.zscore(random_sample)

    # Lets find outliers and to remove the outliers from actual list
    random_sample_outliers = []
    random_sample_without_outliers = []
    for index, value in enumerate(z_scores):
        if abs(value) > 2 or abs(value) < -2:
            random_sample_outliers.append(random_sample[index])
        else:
            random_sample_without_outliers.append(random_sample[index])

    # Lets measure skewness before and after removing the outliers
    skewness_before_removing_outliers = scipy.stats.skew(random_sample)
    skewness_after_removing_outliers = scipy.stats.skew(random_sample_without_outliers)
    skewness_change = skewness_before_removing_outliers - skewness_after_removing_outliers

    # Returns the output list
    report = {
        'actual_list': random_sample,
        'outliers': random_sample_outliers,
        'random_sample_without_outliers': random_sample_without_outliers,
        'skewness_before_removing_outliers': skewness_before_removing_outliers,
        'skewness_after_removing_outliers': skewness_after_removing_outliers,
        'change_in_skewness': skewness_change
    }
    return report

# Lets try box plot to identify and remove the outliers

def remove_outliers_barplots(random_sample):
    """
    A function to identify and remove the outliers from a random_sample using Box Plot
    :param random_sample: Input list to identify and remove outliers to improve skewness
    :return: Output list after removing the outliers and skewness
    """
    # Lets plot the box plot before removing the outliers
    plt.boxplot(random_sample)
    plt.xlabel('Initial Box Plot')
    plt.title('Box Plot Before Removing Outliers')
    initial_boxplot = plt.show()
    # Box plot showed that the outliers started with min value close to 4000 and max value close to 10000.

    # Lets get the data elements in sample below 4000 and calculate how many how many outliers needs to be removed
    initial_random_sample_without_outliers = list(filter(lambda x: (x < 4000), random_sample))
    count_initial_outliers = len(random_sample) - len(initial_random_sample_without_outliers)
    # count_initial_outliers is 9, its not good practice to remove 9 data elements from the sample of 61 data elements

    # Lets try to check another alternative
    final_random_sample_without_outliers = list(filter(lambda x: (x < 7000), random_sample))
    count_final_outliers = len(random_sample) - len(final_random_sample_without_outliers)
    # count_final_outliers is 6, final_boxplot seems decent when compared with previous trail. However, there are
    # still few outliers in the box plot so lets calculate the skewness to make sure whether it is ideal or not

    # Lets plot the box plot after removing the outliers
    plt.boxplot(final_random_sample_without_outliers)
    plt.xlabel('Final Box Plot')
    plt.title('Box Plot After Removing Outliers')
    final_boxplot = plt.show()

    # Lets identify and remove the outliers and measure skewness before and after removing the outliers
    outliers = list(filter(lambda x: (x not in final_random_sample_without_outliers), random_sample))
    skewness_before_removing_outliers = scipy.stats.skew(random_sample)
    skewness_after_removing_outliers = scipy.stats.skew(final_random_sample_without_outliers)
    skewness_change = skewness_before_removing_outliers - skewness_after_removing_outliers

    # Lets return the output list
    report = {
        'initial_list': random_sample,
        'outliers': outliers,
        'list_after_removing_outliers': final_random_sample_without_outliers,
        'skewness_before_removing_outliers': skewness_before_removing_outliers,
        'skewness_after_removing_outliers': skewness_after_removing_outliers,
        'reduced_skewness': skewness_change,
        'intial_boxplot': initial_boxplot,
        'final_boxplot': final_boxplot
    }
    return report


def remove_outliers_IQR(random_sample):
    """
    A function to identify and remove the outliers from a random_sample using IQR
    :param random_sample: input list to identify and remove outliers to improve skewness
    :return: Output list after removing the outliers and skewness
    """
    # Lets calculate the threshold of outliers using IQR
    median_random_sample = scipy.median(random_sample)
    first_half_data = list(filter(lambda x: (x < median_random_sample), random_sample))
    second_half_data = list(filter(lambda x: (x > median_random_sample), random_sample))
    q1 = scipy.median(first_half_data)
    q3 = scipy.median(second_half_data)
    iqr_sample_data = q3 - q1
    lower_bound = q1 - 1.5 * iqr_sample_data
    higher_bound = q3 + 1.5 * iqr_sample_data

    # Lets plot the box plot before removing the outliers
    plt.boxplot(random_sample)
    plt.xlabel('Initial BoxPlot')
    plt.title('BoxPlot Before Removing Outliers')
    initial_boxplot = plt.show()

    # Lets identify and remove the outliers and measure skewness before and after removing the outliers
    outliers = list(filter(lambda x: (x > higher_bound or x < lower_bound), random_sample))
    random_sample_without_outliers = list(filter(lambda x: (x not in outliers), random_sample))
    skewness_before_removing_outliers = scipy.stats.skew(random_sample)
    skewness_after_removing_outliers = scipy.stats.skew(random_sample_without_outliers)
    skewness_change = skewness_before_removing_outliers - skewness_after_removing_outliers

    # Lets plot the box plot before removing the outliers
    plt.boxplot(random_sample_without_outliers)
    plt.xlabel('Final BoxPlot')
    plt.title('BoxPlot After Removing Outliers Using IQR')
    final_boxplot = plt.show()

    # Lets return the output list
    report = {
        'initial_list': random_sample,
        'outliers': outliers,
        'list_after_removing_outliers': random_sample_without_outliers,
        'skewness_before_removing_outliers': skewness_before_removing_outliers,
        'skewness_after_removing_outliers': skewness_after_removing_outliers,
        'reduced_skewness': skewness_change,
        'intial_boxplot': initial_boxplot,
        'final_boxplot': final_boxplot
    }
    return report



salary = [0, 10, 20, 200, 300, 400, 410, 450, 460, 500, 600, 700, 710, 712, 713, 745, 764, 1000, 1010, 1020, 1030,
          1040, 1111, 1234, 1300, 1338, 1670, 1680, 1690, 1635, 1645, 1655, 1665, 1700, 1710, 2000, 2020, 2030, 2020,
          2030, 2010, 2050, 2060, 2010, 2030, 2222, 2134, 2020, 2030, 2050, 2050, 3000, 4050, 5000, 6000, 7000, 8010,
          9010, 9020, 10000, 10020]

report1 = remove_outliers_z_score(salary)
report2 = remove_outliers_barplots(salary)
report3 = remove_outliers_IQR(salary)
print(report1)
print(report2)
print(report3)
