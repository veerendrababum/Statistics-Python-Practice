__author__ = 'Veerendra'
__date__ = '30-June-2020'


'''
We have taken a data sample of salaries in a street and our end goal is to find out the good estimate average salary.
We need to check the skewness in the sample and if there is skewness then remove the outliers to reduce the skewness
Removing skewness helps to get accurate results on the data....

Multiple approaches to find skewness/outliers
    :Method 1: Z score
    :Method 2: box plot
    :Method 3: std
    :Method 4: Gaussian 
    :Method 5: IQR
'''

import scipy.stats


def remove_outliers(x):
    """
    Method 1: is generally used when the expected data is normally distributed
    :param x: input list to remove skewness
    :return:  output list after removing skewness
    """

    skewness_before_removing_outliers = scipy.stats.skew(x)
    z_score_x = scipy.stats.zscore(x)
    outliers = []
    x_with_out_outliers = []
    for index, value in enumerate(z_score_x):
        if abs(value) > 2:
            outliers.append(x[index])
        else:
            x_with_out_outliers.append(x[index])

    skewness_after_removing_outliers = scipy.stats.skew(x_with_out_outliers)

    report = {
        'initial_list': x,
        'initial_skewness': skewness_before_removing_outliers,
        'outliers': outliers,
        'final_list': x_with_out_outliers,
        'final_skewness': skewness_after_removing_outliers
    }
    return report


salary = [0, 10, 20, 200, 300, 400, 410, 450, 460, 500, 600, 700, 710, 712, 713, 745, 764, 1000, 1010, 1020, 1030, 1040,
          1111, 1234, 1300, 1338, 1670, 1680, 1690, 1635, 1645, 1655, 1665, 1700, 1710, 2000, 2020, 2030, 2020, 2030,
          2010, 2050, 2060, 2010, 2030, 2222, 2134, 2020, 2030, 2050, 2050, 3000, 4050, 5000, 6000, 7000, 8010, 9010,
          9020, 10000, 10020]

report = remove_outliers(salary)
print(report['outliers'])
