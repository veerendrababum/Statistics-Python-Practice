__author__ = 'Veerendra'
__date__ = '14-July-2020'
'''
Problem statement: The national average salary is $56,516. I assumed that my community size of 1000 people's average 
salary is lesser than the national average salary. To check my hypothesis I took a sample of ten random people salary

Goal is to write the code to find out the best test(ztestvsttest) to perform the hypothesis

Null hypothesis: Average salary of community is same as national average salary
Alternate hypothesis: Average salary of community is lesser than national average salary
 
'''
import numpy as np
import scipy
import scipy.stats


class ztestVsttest():
    def __init__(self, random_sample):
        self.random_sample = random_sample
        self.output = {}

        # Lets check whether n>30 or n<30 to decide the test
        random_sample_size = len(self.random_sample)
        if random_sample_size > 30:
            self.ztest()
        else:
            self.ttest()

    def ztest(self):
        '''
        Function to perform hypothesis by finding out the p value using z stat
        '''

        random_sample = self.random_sample

        # Lets find out the Z stat
        random_sample_size = len(self.random_sample)
        random_sample_mean = scipy.mean(random_sample)
        random_sample_std = scipy.std(random_sample)
        population_mean = 56516 # The national average salary is 56,516
        standard_error = random_sample_std / scipy.sqrt(random_sample_size)
        z_stat = (random_sample_mean - population_mean) / standard_error

        # Lets find out the p-value
        pval = scipy.stats.norm.sf(abs(z_stat)) #one-sided

        # Perform the hypothesis using the p value of z stat
        if pval < 0.05:  # alpha value is 0.05 or 5%
            print("Performed hypothesis testing using ztest. We are rejecting null hypothesis since pval "
                      "< significance level")
        else:
            print("Performed hypothesis testing using ztest. We cant reject the null hypothesis since pval "
                      "> significance level")

        report = {
            'z_statistic': z_stat,
            'pvalue': pval
        }

        self.output = report

        return report

    def ttest(self):
        '''
        Function to perform hypothesis by finding out the p value using t stat
        '''
        random_sample = self.random_sample

        # Check if the distribution has any outliers
        z_scores = scipy.stats.zscore(random_sample)
        outliers = list(filter(lambda x: (x > 3 or x < -3), z_scores))

        # t stat will be calculated only if the sample distribution has no outliers
        if outliers == []:
            # No outliers, lets find out the t stat
            random_sample_size = len(self.random_sample)
            random_sample_mean = scipy.mean(random_sample)
            random_sample_std = scipy.std(random_sample)
            population_mean = 56516# The national average salary is 56,516
            standard_error = random_sample_std / scipy.sqrt(random_sample_size)
            t_stat = (random_sample_mean - population_mean) / standard_error

            # Lets find out the p-value
            pval = scipy.stats.t.sf(np.abs(t_stat), random_sample_size - 1)

            # Perform the hypothesis using the p value of t stat
            if pval < 0.05:  # alpha value is 0.05 or 5%
                print("Performed hypothesis testing using ttest. We are rejecting null hypothesis since pval "
                      "< significance level")
            else:
                print("Performed hypothesis testing using ttest. We cant reject the null hypothesis since pval "
                      "> significance level")

        else:
            print('We can\'t calculate the tstat since we have outliers in the distribution')

        report = {
            't_statistic': t_stat,
            'pvalue': pval
        }

        self.output = report

        return report



salary = [21675, 34345, 37673, 23000, 24675, 25673, 25977, 27890, 85999, 34000]

check_hypothesis= ztestVsttest(salary)
print(check_hypothesis.output)
