import random
from helpers.data_manipulation import r2_score


class Chromossome:
    def __init__(self, genes: str):
        self.genes = genes
        self.fit = self.fitness()

    def value(self):
        return self.genes

    def fitness(self, features, y):
        # implement fitness based on KNN return
        return 0.0

    def mutation(self, mutation_rate: float):
        genes_len = len(self.genes)

        for i in range(genes_len):
            if random.random() > mutation_rate:
                continue
            self.genes[i] = '1' if self.genes[i] == '0' else '0'

        self.fit = self.fitness()

        return self
