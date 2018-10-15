

import random
import sys

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
sys.setrecursionlimit(10000)

class Regression:
    def __init__(self):
        self.k = 5
        self.metric = np.mean
        self.kdtree = None
        self.houses = None
        self.values = None

    def set_data(self, houses, values):
        """
        Sets houses and values data
        :param houses: pandas.DataFrame with houses parameters
        :param values: pandas.Series with houses values
        """
        self.houses = houses
        self.values = values
        self.kdtree = KDTree(self.houses)

    def regress(self, query_point):
        distances, indexes = self.kdtree.query(query_point, self.k)
        m = self.metric(self.values.iloc[indexes])
        if np.isnan(m):
            zomg
        else:
            return m
    
    def error_rate(self, folds):
        holdout = 1 / float(folds)
        errors = []
        for fold in range(folds):
            y_hat, y_true = self.__validation_data(holdout)
            errors.append(mean_absolute_error(y_true, y_hat))
        return errors
        
    def __validation_data(self, holdout):
        test_rows = random.sample(self.df.index, int(round(len(self.df) * holdout)))
        train_rows = set(range(len(self.df))) - set(test_rows)
        df_test = self.df.ix[test_rows]
        df_train = self.df.drop(test_rows)
        test_values = self.values.ix[test_rows]
        train_values = self.values.ix[train_rows]
        kd = Regression(data=df_train, values=train_values)

        y_hat = []
        y_actual = []
            
        for idx, row in df_test.iterrows():
            y_hat.append(kd.regress(row))
            y_actual.append(self.values[idx])
        return (y_hat, y_actual)

    def plot_error_rates(self):
        folds = range(2, 11)
        errors = pd.DataFrame({'max': 0, 'min': 0}, idnex=folds)
        for i in folds:
            error_rates = r.error_rate(f)
            errors['max'][f] = max(error_rates)
            errors['min'][f] = min(error_rates)
            errors.plot(title="Mean Absolute Error of KNN over different folds")
            plt.show()

class RegressionTest(object):
  """
  Take in King County housing data, calculate and plot the kNN regression error rate.
  """

  def __init__(self):
    self.houses = None
    self.values = None

  def load_csv_file(self, csv_file, limit=None):
    """
    Loads CSV file with houses data
    :param csv_file: CSV file name
    :param limit: number of rows of file to read
    """
    houses = pd.read_csv(csv_file, nrows=limit)
    self.values = houses['AppraisedValue']
    houses = houses.drop('AppraisedValue', 1)
    houses = (houses - houses.mean()) / (houses.max() - houses.min())
    self.houses = houses
    self.houses = self.houses[['lat', 'long', 'SqFtLot']]

  def plot_error_rates(self):
    """
    Plots MAE vs #folds
    """
    folds_range = range(2, 11)
    errors_df = pd.DataFrame({'max': 0, 'min': 0}, index=folds_range)
    for folds in folds_range:
      errors = self.tests(folds)
      errors_df['max'][folds] = max(errors)
      errors_df['min'][folds] = min(errors)
    errors_df.plot(title='Mean Absolute Error of KNN over different folds_range')
    plt.xlabel('#folds_range')
    plt.ylabel('MAE')
    plt.show()

  def tests(self, folds):
    """
    Calculates mean absolute errors for series of tests
    :param folds: how many times split the data
    :return: list of error values
    """
    holdout = 1 / float(folds)
    errors = []
    for _ in range(folds):
      values_regress, values_actual = self.test_regression(holdout)
      errors.append(mean_absolute_error(values_actual, values_regress))

    return errors

  def test_regression(self, holdout):
    """
    Calculates regression for out-of-sample data
    :param holdout: part of the data for testing [0,1]
    :return: tuple(y_regression, values_actual)
    """
    test_rows = random.sample(self.houses.index.tolist(),
                  int(round(len(self.houses) * holdout)))
    train_rows = set(range(len(self.houses))) - set(test_rows)
    df_test = self.houses.ix[test_rows]
    df_train = self.houses.drop(test_rows)

    train_values = self.values.ix[train_rows]
    regression = Regression()
    regression.set_data(houses=df_train, values=train_values)

    values_regr = []
    values_actual = []

    for idx, row in df_test.iterrows():
      values_regr.append(regression.regress(row))
      values_actual.append(self.values[idx])

    return values_regr, values_actual


def main():
  regression_test = RegressionTest()
  regression_test.load_csv_file('king_county_data_geocoded.csv', 100)
  regression_test.plot_error_rates()


if __name__ == '__main__':
  main()