"""
outliers: multiple ways
1. IQr
2. z
3.
4.
5.
"""
import scipy.stats
import matplotlib.pyplot as plt


class OutliersDetection:
    def __init__(self, random_sample, method='z'):
        self.input = random_sample
        self.output = {}
        if method == 'z':
            self.z_score_evaluation()
        elif method.lower() == 'iqr':
            self.iqr_evaluation()
        else:
            self.intelligent_model()


    def z_score_evaluation(self):
        random_sample = self.input
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
            'reduced_skewness': skewness_change
        }
        self.output = report
        return report

    def iqr_evaluation(self):
        random_sample = self.input
        median_random_sample = scipy.median(random_sample)
        first_half_data = list(filter(lambda x: (x < median_random_sample), random_sample))
        second_half_data = list(filter(lambda x: (x > median_random_sample), random_sample))
        q1 = scipy.median(first_half_data)
        q3 = scipy.median(second_half_data)
        iqr_sample_data = q3 - q1
        lower_bound = q1 - 1.5 * iqr_sample_data
        higher_bound = q3 + 1.5 * iqr_sample_data

        # Lets plot the box plot before removing the outliers
        # plt.boxplot(random_sample)
        # plt.xlabel('Initial BoxPlot')
        # plt.title('BoxPlot Before Removing Outliers')
        # initial_boxplot = plt.show()

        # Lets identify and remove the outliers and measure skewness before and after removing the outliers
        outliers = list(filter(lambda x: (x > higher_bound or x < lower_bound), random_sample))
        random_sample_without_outliers = list(filter(lambda x: (x not in outliers), random_sample))
        skewness_before_removing_outliers = scipy.stats.skew(random_sample)
        skewness_after_removing_outliers = scipy.stats.skew(random_sample_without_outliers)
        skewness_change = skewness_before_removing_outliers - skewness_after_removing_outliers

        # Lets plot the box plot before removing the outliers
        # plt.boxplot(random_sample_without_outliers)
        # plt.xlabel('Final BoxPlot')
        # plt.title('BoxPlot After Removing Outliers Using IQR')
        # final_boxplot = plt.show()

        # Lets return the output list
        report = {
            'initial_list': random_sample,
            'outliers': outliers,
            'list_after_removing_outliers': random_sample_without_outliers,
            'skewness_before_removing_outliers': skewness_before_removing_outliers,
            'skewness_after_removing_outliers': skewness_after_removing_outliers,
            'reduced_skewness': skewness_change,
            # 'intial_boxplot': initial_boxplot,
            # 'final_boxplot': final_boxplot
        }
        self.output = report
        return report

    def intelligent_model(self):
        report = self.iqr_evaluation()
        second_report = self.z_score_evaluation()
        if report['reduced_skewness'] > second_report['reduced_skewness']:
            self.output = second_report
        else:
            self.output = report




salary = [0, 10, 20, 200, 300, 400, 410, 450, 460, 500, 600, 700, 710, 712, 713, 745, 764, 1000, 1010, 1020, 1030,
          1040, 1111, 1234, 1300, 1338, 1670, 1680, 1690, 1635, 1645, 1655, 1665, 1700, 1710, 2000, 2020, 2030, 2020,
          2030, 2010, 2050, 2060, 2010, 2030, 2222, 2134, 2020, 2030, 2050, 2050, 3000, 4050, 5000, 6000, 7000, 8010,
          9010, 9020, 10000, 10020]

outliers = OutliersDetection(salary, method='intelligent')
report = outliers.z_score_evaluation()
# print(outliers.output)
print(report)