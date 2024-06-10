import numpy as np

from vangogh.population import Population


def select(population, selection_size, generation, selection_name):
    if 'tournament' in selection_name:
        tournament_size = int(selection_name.split('_')[-1])
        return tournament_select(population, selection_size, tournament_size)
    
    elif 'roulette_wheel_selection' in selection_name:
        return roulette_wheel_selection(population, selection_size)
    
    elif 'stochastic_universal_sampling' in selection_name:
        return stochastic_universal_sampling(population, selection_size)

    elif 'linear_rank_selection' in selection_name:
        return linear_rank_selection(population, selection_size)

    elif 'optimization' in selection_name:
         return run_optimization(population, selection_size, generation, selection_name)       

    else:
        raise ValueError('Invalid selection name:', selection_name)


# ----------------- Begin selection methods -----------------
# Here we define the possible selection methods
# From these we will pick certain methods according to a situation
#------------------------------------------------------------

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


def roulette_wheel_selection(population, selection_size):
    genotype_length = population.genes.shape[1]
    selected = Population(selection_size, genotype_length, "N/A")

    # Calculate the total fitness, inverse it to minimize
    total_inverse_fitness = np.sum(1 / population.fitnesses)

    # Calculate the selection probablility of every individual
    selection_probabilities = (1 / population.fitnesses) / total_inverse_fitness

    # Loop trough the population looking for good picks by increasing the selection probability of individuals
    for i in range(selection_size):
        random_number = np.random.uniform(0, 1)        
        iSum = 0
        j = 0
    
        while iSum < random_number and j < len(selection_probabilities):
            iSum += selection_probabilities[j]
            j += 1

        selected.genes[i] = population.genes[j-1]
        selected.fitnesses[i] = population.fitnesses[j-1]

    return selected


def stochastic_universal_sampling(population, selection_size):
    genotype_length = population.genes.shape[1]
    selected = Population(selection_size, genotype_length, "N/A")

    inverted_fitnesses = 1.0 / population.fitnesses

    total_fitness = np.sum(inverted_fitnesses)
    pointer_distance = total_fitness / selection_size

    start_point = np.random.uniform(0, pointer_distance)
    pointers = [start_point + i * pointer_distance for i in range(selection_size)]

    fitness_sum = 0
    current_member = 0
    for i in range(selection_size):
        # Move the fitness sum to the pointer
        while fitness_sum < pointers[i]:
            fitness_sum += inverted_fitnesses[current_member]
            current_member += 1
            
        # Select the individual
        selected.genes[i] = population.genes[current_member - 1]
        selected.fitnesses[i] = population.fitnesses[current_member - 1]

    return selected


def linear_rank_selection(population, selection_size):
    genotype_length = population.genes.shape[1]
    selected = Population(selection_size, genotype_length, "N/A")

    # Rank individuals based on their fitnesses (smallest fitness has rank 1)
    sorted_indices = np.argsort(population.fitnesses)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(population.fitnesses), 0, -1)  # Highest rank for smallest fitness

    # Calculate the sum v
    n = len(population.fitnesses)
    v = 1 / (n - 2.001)

    # Calculate the probabilities based on ranks
    probabilities = ranks / (n * (n - 1))

    # Select individuals based on their rank probabilities
    for i in range(selection_size):
        alpha = np.random.uniform(0, v)
        cumulative_sum = 0
        for j in range(n):
            cumulative_sum += probabilities[j]
            if alpha <= cumulative_sum:
                selected.genes[i] = population.genes[sorted_indices[j]]
                selected.fitnesses[i] = population.fitnesses[sorted_indices[j]]
                break

    

    return selected

# ------------------ End selection methods ------------------


# Calculating the euclidean distance between i-th and j-th individuals
def calculate_euclidean_distance(individual1, individual2):
    return np.linalg.norm(individual1 - individual2)


# Calculating the diversity
def calculate_diversity(population, generation, max_distance):
    n = len(population.genes)
    total_distance = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            distance = calculate_euclidean_distance(population.genes[i], population.genes[j])
            total_distance += distance

    average_distance = total_distance / (n * (n - 1) / 2)
    diversity = (1 / max_distance) * (1 / np.log(n - 1)) * average_distance
    return diversity


def get_fitness_statistics(population):
    f_max = np.max(population.fitnesses)
    f_min = np.min(population.fitnesses)
    f_best = np.min(population.fitnesses) # Minimize
    return f_max, f_min, f_best

# Calculating the quality here
def calculate_quality(population):
    f_max, f_best, f_min = get_fitness_statistics(population)
    return f_best / np.sqrt(f_max ** 2 + f_min ** 2)


# Calculate combined criterion
def calculate_combined_criterion(diversity, quality, generation):
    t = generation + 1  # Assuming generation starts from 0
    combined_criterion = (1 / t) * diversity + ((t - 1) / t) * quality
    return combined_criterion


def run_optimization(population, selection_size, generation, selection_name="linear_rank_selection"):
    selection_methods = ["tournament_4", "stochastic_universal_sampling", "roulette_wheel_selection"]
    compare_methods = []

    for methods in selection_methods:
        selected_population = select(population, selection_size, generation, methods)
    
        # Calculate statistics
        diversity = calculate_diversity(selected_population, generation, max_distance=100)  # Adjust max_distance as needed
        quality = calculate_quality(selected_population)
        combined_criterion = calculate_combined_criterion(diversity, quality, generation)
    
        compare_methods.append({'method':methods, 'selected_population':selected_population, 'diversity':diversity, 'quality':quality, 'combined_criterion':combined_criterion})

   
    # ADD some logic to run multiple selection strategies and then boom bats pick one :)
    best_method = max(compare_methods, key=lambda x: x['combined_criterion'])

    # Update the population for the next generation

    selected = best_method['selected_population']
    diversity = best_method['diversity']
    quality = best_method['quality']
    combined_criterion = best_method['combined_criterion']

    print(f"Best current method: {best_method['method']}, Diversity={diversity}, Quality={quality}, Combined Criterion={combined_criterion}")

    return selected

