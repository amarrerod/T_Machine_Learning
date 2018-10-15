
# Comparacion de mi implementacion con el algoritmo KNN de scikit learn

from sklearn.neighbors import KNeighborsRegressor
import random
import sys
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)

class KNNRegression:
    def __init__(self, k = None):
        if (k is None):
            raise ValueError("K must be given")
        else:
            self.k = k
            self.kdtree = None
            self.rows = None
            self.values = None

    def load_csv_file(self, csv_file, value, limit=None):
        rows = pd.read_csv(csv_file, nrows=limit)
        self.values = rows[value]
        rows = rows.drop(value, 1)
        rows = (rows - rows.mean()) / (rows.max() - rows.min())
        self.rows = rows
        self.rows = self.rows[['Duration', 'Bike ID']]
        self.kdtree = KDTree(self.rows)

    
    def regress(self, query_row):
        duration, idxs = self.kdtree.query(query_row, self.k)
        m = np.mean(self.values.iloc[idxs])
        if np.isnan(m):
            zomg
        else:
            return m

    def error_rate(self, folds):
        holdout = 1 / float(folds)
        errors = []
        for fold in range(folds):
            y_hat, y_true = self.__validation(holdout)
            errors.append(mean_absolute_error(y_true, y_hat))
        return errors
    
    def __validation(self, holdout):
        test_rows = random.sample(self.rows.index, int(round(len(self.rows) * holdout)))
        train_rows = set(range(len(self.rows))) - set(test_rows)
        df_test = self.rows.ix[test_rows]
        df_train = self.rows.drop(test_rows)
        test_values = self.values.ix[test_rows]
        train_values = self.values.ix[train_rows]
        kd = Regression(self.k)
        kd.rows = df_train
        kd.values = train_values
        y_hat, y_actual = [], []
        for idx, row in df_test.iterrows():
            y_hat.append(kd.regress(row))
            y_actual.append(self.values[idx])
        return (y_hat, y_actual)

    def plot_error_rates(self):
        folds = range(2, 5)
        errors = pd.DataFrame({'max': 0, 'min': 0}, index = folds)
        for f in folds:
            error_rates = r.error_rate(f)
            errors['max'][f] = max(error_rates)
            errors['min'][f] = min(error_rates)
        errors.plot(title = "Mean Absolute Error of KNN over different folds")
        plt.show()

def main():
        knn = KNNRegression(3)
        knn.load_csv_file('king_county_data_geocoded', 200)
        knn.plot_error_rates()

if __name__ == '__main__':
  main()

# Doing the comparison