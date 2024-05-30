import numpy as np


def crossover(genes, method="ONE_POINT"):
    parents_1 = np.vstack((genes[:len(genes) // 2], genes[:len(genes) // 2]))
    parents_2 = np.vstack((genes[len(genes) // 2:], genes[len(genes) // 2:]))

    if method == "ONE_POINT":
        crossover_points = np.random.randint(0, genes.shape[1], size=genes.shape[0])
        offspring = np.zeros(shape=genes.shape, dtype=int)
        for i in range(len(genes)):
            offspring[i,:] = np.where(np.arange(genes.shape[1]) <= crossover_points[i], parents_1[i,:], parents_2[i,:])

    elif method == "TWO_POINT":
        crossover_points = np.sort(np.random.choice(genes.shape[1], size=(genes.shape[0], 2), replace=False))
        offspring = np.zeros(shape=genes.shape, dtype=int)
        for i in range(len(genes)):
            start, end = crossover_points[i]
            offspring[i, start:end] = parents_1[i, start:end]
            offspring[i, :start] = parents_2[i, :start]
            offspring[i, end:] = parents_2[i, end:]

    elif method == "UNIFORM":
        crossover_points = np.random.uniform(0, 1, size=genes.shape).round().astype(int)
        offspring = np.zeros(shape=genes.shape, dtype=int)
        for i in range(len(genes)):
            for j in range(genes.shape[1]):
                if crossover_points[i, j] == 1:
                    offspring[i, j] = parents_1[i, j]
                else:
                    offspring[i, j] = parents_2[i, j]

    else:
        raise Exception("Unknown crossover method")

    return offspring


def mutate(genes, feature_intervals,
           mutation_probability=0.1, num_features_mutation_strength=0.05):
    mask_mut = np.random.choice([True, False], size=genes.shape,
                                p=[mutation_probability, 1 - mutation_probability])

    mutations = generate_plausible_mutations(genes, feature_intervals,
                                             num_features_mutation_strength)

    offspring = np.where(mask_mut, mutations, genes)

    return offspring


def generate_plausible_mutations(genes, feature_intervals,
                                 num_features_mutation_strength=0.25):
    mutations = np.zeros(shape=genes.shape)

    for i in range(genes.shape[1]):
        range_num = feature_intervals[i][1] - feature_intervals[i][0]
        low = -num_features_mutation_strength / 2
        high = +num_features_mutation_strength / 2

        mutations[:, i] = range_num * np.random.uniform(low=low, high=high,
                                                        size=mutations.shape[0])
        mutations[:, i] += genes[:, i]

        # Fix out-of-range
        mutations[:, i] = np.where(mutations[:, i] > feature_intervals[i][1],
                                   feature_intervals[i][1], mutations[:, i])
        mutations[:, i] = np.where(mutations[:, i] < feature_intervals[i][0],
                                   feature_intervals[i][0], mutations[:, i])

    mutations = mutations.astype(int)
    return mutations
