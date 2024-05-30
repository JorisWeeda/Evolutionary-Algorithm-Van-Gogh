import numpy as np

from vangogh.population import Population


def select(population, selection_size, selection_name='tournament_4'):
    if 'tournament' in selection_name:
        tournament_size = int(selection_name.split('_')[-1])
        return tournament_select(population, selection_size, tournament_size)
    elif 'roulette_wheel_selection' in selection_name:
        roulette_wheel_selection(population)
    else:
        raise ValueError('Invalid selection name:', selection_name)


def one_tournament_round(population, tournament_size, return_winner_index=False):
    rand_perm = np.random.permutation(len(population.fitnesses))
    competing_fitnesses = population.fitnesses[rand_perm[:tournament_size]]
    winning_index = rand_perm[np.argmin(competing_fitnesses)]
    if return_winner_index:
        return winning_index
    else:
        return {
            'genotype': population.genes[winning_index, :],
            'fitness': population.fitnesses[winning_index],
        }


def tournament_select(population, selection_size, tournament_size=4):
    genotype_length = population.genes.shape[1]
    selected = Population(selection_size, genotype_length, "N/A")

    n = len(population.fitnesses)
    num_selected_per_iteration = n // tournament_size
    num_parses = selection_size // num_selected_per_iteration

    for i in range(num_parses):
        # shuffle
        population.shuffle()

        winning_indices = np.argmin(population.fitnesses.squeeze().reshape((-1, tournament_size)),
                                    axis=1)
        winning_indices += np.arange(0, n, tournament_size)

        selected.genes[i * num_selected_per_iteration:(i + 1) * num_selected_per_iteration,
        :] = population.genes[winning_indices, :]
        selected.fitnesses[i * num_selected_per_iteration:(i + 1) * num_selected_per_iteration] = \
        population.fitnesses[winning_indices]

    return selected

def roulette_wheel_selection(population):
    fitness = population.fitnesses
    total_fitness = np.sum(fitness)
    selected_individuals = []


    for i in range(len(population.fitnesses)):
        random_number = np.random.randint(low=0, high=total_fitness) # Might be an issue here!
        iSum = 0
        j = 0

        while iSum < random_number and j < len(population.fitnesses):
            iSum += population.fitnesses[i]
            j += 1
        
        selected_individuals.append(population[j-1])
    
    return selected_individuals


def stochastic_universalsampling(population):
    pass

def linear_rank_selection(population):
    pass

def exponential_rank_selection(population):
    pass

def truncation_selection(population):
    pass

