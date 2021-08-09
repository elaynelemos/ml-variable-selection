from pandas import DataFrame
from genetic_algorithm.models.population import Population


def genetic_algorithm(
            population_size:int,
            max_generations: int,
            mutation_rate:float,
            database: DataFrame):

    population = Population(population_size, mutation_rate, database)
    generations = []

    while len(generations) < max_generations and not kept_fit(generations):
        generations.append(population)
        print_generation(len(generations), population)

        population.reproduction()

    generations.append(population)
    print_generation(len(generations), population)
    solution = generations[-1].best_fit().value()
    print(f'Solution: f({solution}) = {generations[-1].best_fit().fit}')

    return solution


def kept_fit(generations):
    average_fits = []
    best_fits = []
    target_error = 1e-4
    min_generations = 5

    if len(generations) < min_generations:
        return False

    for generation in generations:
        average_fits.append(generation.average_fit())
        best_fits.append(generation.best_fit())

    init = len(generations) - min_generations + 1
    average = average_fits[init]
    best = best_fits[init]
    for i in range(init + 1, len(average_fits)):
        if abs(average_fits[i] - average) > target_error:
            average = None

        if abs(best_fits[i] - best) > target_error:
            best = None

        if average is None and best is None:
            return False

    return True


def print_generation(gen_index, generation):
    chromossomes_genes = []
    chromossomes = generation.chromossomes

    for i in range(len(chromossomes)):
        chromossomes_genes.append(
            f"Chromossome {i + 1} '{chromossomes[i].genes}' = "
            f'{chromossomes[i].value()}')

    output = '\n'.join(chromossomes_genes)
    print(f'Generation #{gen_index}')
    print(output)
