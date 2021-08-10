import random
from pandas.core.frame import DataFrame
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from helpers.data_manipulation import filter_database


class Chromossome:
    def __init__(self, genes: str, X_train, X_test, y_train, y_test):
        self.genes = genes
        self.fit = self.fitness(
            filter_database(X_train, genes),
            filter_database(X_test, genes),
            y_train,
            y_test
        )

    def value(self):
        return self.genes

    def fitness(self, X_train, X_test, y_train, y_test):
        classifier = KNeighborsRegressor(3).fit(X_train, y_train.values.ravel())
        y_predict = classifier.predict(X_test)

        return r2_score(y_true=y_test.values.ravel(), y_pred=y_predict)

    def mutation(self, mutation_rate: float, X_train, X_test, y_train, y_test):
        genes_len = len(self.genes)

        for i in range(genes_len):
            if random.random() > mutation_rate:
                continue
            self.genes[i] = '1' if self.genes[i] == '0' else '0'

        X_train = filter_database(X_train, self.genes)
        X_test = filter_database(X_test, self.genes)
        self.fit = self.fitness(X_train, X_test, y_train, y_test)

        return self
