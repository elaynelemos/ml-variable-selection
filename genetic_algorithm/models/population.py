import random
from pandas import DataFrame
from genetic_algorithm.models.chromossome import Chromossome
from helpers.data_manipulation import prepare_database


class Population:
    def __init__(self, population_size: int, mutation_rate: int, database: DataFrame):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.X_train, self.X_test, self.y_train, self.y_test = prepare_database(database)
        self.chromossomes = self.generate_initial_chromossomes(
            population_size,
            database.shape[1] - 1
        )


    def create_random_genes(self, X_size: int):
        positions = range(X_size)
        replacements = random.sample(positions, random.randint(3, 12))
        genes = ''.join(('0' if i not in replacements else '1') for i in positions)

        return genes


    def generate_initial_chromossomes(self, population_size: int, X_size: int):
        population  = [ Chromossome(
            self.create_random_genes(X_size), self.X_train, self.X_test,
            self.y_train, self.y_test
        ) for _ in range(population_size) ]

        return population


    def best_fit(self):
        best_chromossome = self.chromossomes[0]
        for chromossome in self.chromossomes:
            if chromossome.fit < best_chromossome.fit:
                best_chromossome = chromossome

        return best_chromossome


    def worst_fit(self):
        worst_chromossome = self.chromossomes[0]
        for chromossome in self.chromossomes:
            if chromossome.fit > worst_chromossome.fit:
                worst_chromossome = chromossome

        return worst_chromossome


    def average_fit(self):
        total_fit = 0
        for chromossome in self.chromossomes:
            total_fit += chromossome.fit

        return total_fit/self.population_size


    def tournment(self):
        index = random.sample(range(self.population_size), 2)
        parents = [self.chromossomes[index[0]], self.chromossomes[index[1]]]

        if parents[0].fit < parents[1].fit:
            return index[0]

        return index[1]


    def crossover(self):
        cross_index = 0.85
        parents_index = []
        while len(parents_index) < 2:
            index = self.tournment()
            if index not in parents_index:
                parents_index.append(index)

        parents = [
            self.chromossomes[parents_index[0]],
            self.chromossomes[parents_index[1]],
        ]
        if random.random() > cross_index:
            return parents

        children = ['', '']
        children[0] = parents[0].genes
        children[1] = parents[1].genes
        children[0] = ''.join((parents[1].genes[i] if i%2 == 1 else children[0][i]) for i in range(len(children[0])))
        children[1] = ''.join((parents[0].genes[i] if i%2 == 1 else children[1][i]) for i in range(len(children[1])))

        children[0] = Chromossome(
            children[0], self.X_train, self.X_test, self.y_train, self.y_test
        ).mutation(self.mutation_rate,
            self.X_train, self.X_test, self.y_train, self.y_test)
        children[1] = Chromossome(
            children[1], self.X_train, self.X_test, self.y_train, self.y_test
        ).mutation(self.mutation_rate,
            self.X_train, self.X_test, self.y_train, self.y_test)

        return children


    def reproduction(self):
        best_chromossome = self.best_fit()
        new_chromossomes = []

        while len(new_chromossomes) < self.population_size:
            new_chromossomes += self.crossover()

        new_chromossomes.append(best_chromossome)
        self.chromossomes = new_chromossomes
        self.chromossomes.remove(self.worst_fit())
