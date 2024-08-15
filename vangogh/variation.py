import numpy as np
import torch

def crossover(genes: np.ndarray, method: str = "ONE_POINT", rbfn: torch.nn.Module = None) -> tuple:
    """
    Performs crossover operation on a set of genes using the specified method.

    Parameters:
        genes (np.ndarray): The array of genes (population).
        method (str): The crossover method to use ('ONE_POINT', 'TWO_POINT', 'UNIFORM', or 'RBFNX').
        rbfn (torch.nn.Module, optional): A RBF network model used for 'RBFNX' method.

    Returns:
        tuple: A tuple containing the offspring and the crossover mask.
    """
    # Define parents
    parents_1 = np.vstack((genes[:len(genes) // 2], genes[:len(genes) // 2]))
    parents_2 = np.vstack((genes[len(genes) // 2:], genes[len(genes) // 2:]))

    offspring = np.zeros_like(genes)
    crossover = np.zeros_like(genes)

    if method == "ONE_POINT":
        crossover_points = np.random.randint(0, genes.shape[1], size=genes.shape[0])
        for i in range(len(genes)):
            crossover_point = crossover_points[i]
            offspring[i, :] = np.where(np.arange(genes.shape[1]) <= crossover_point, parents_1[i, :], parents_2[i, :])
            crossover[i, crossover_point] = 1

    elif method == "TWO_POINT":
        crossover_points = np.sort(np.random.choice(genes.shape[1], size=(genes.shape[0], 2), replace=False), axis=1)
        for i in range(len(genes)):
            start, end = crossover_points[i]
            offspring[i, start:end] = parents_1[i, start:end]
            offspring[i, :start] = parents_2[i, :start]
            offspring[i, end:] = parents_2[i, end:]
            crossover[i, start:end] = 1

    elif method == "UNIFORM":
        crossover_points = np.random.uniform(0, 1, size=genes.shape).round().astype(int)
        for i in range(len(genes)):
            for j in range(genes.shape[1]):
                if crossover_points[i, j] == 1:
                    offspring[i, j] = parents_1[i, j]
                else:
                    offspring[i, j] = parents_2[i, j]
                crossover[i, j] = crossover_points[i, j]

    elif method == "RBFNX":
        if rbfn is None:
            raise ValueError("rbfn must be provided for RBFNX method")
        
        rbfn.eval()
        with torch.no_grad():
            y = rbfn.forward(torch.tensor(parents_1, dtype=torch.float32), 
                             torch.tensor(parents_2, dtype=torch.float32))
        y_np = y.numpy()

        r_values = np.random.uniform(0, 1, size=y_np.shape)
        mask = np.where(y_np < r_values, 0, 1)

        offspring = np.where(mask == 1, parents_1, parents_2)
        crossover = mask

    else:
        raise ValueError("Unknown crossover method")

    return offspring, crossover

def mutate(genes: np.ndarray, feature_intervals: list, mutation_probability: float = 0.1, 
           num_features_mutation_strength: float = 0.05) -> np.ndarray:
    """
    Applies mutation to the genes with a given probability.

    Parameters:
        genes (np.ndarray): The array of genes (population).
        feature_intervals (list): A list of tuples specifying the interval for each feature.
        mutation_probability (float): Probability of mutation occurring.
        num_features_mutation_strength (float): The mutation strength for each feature.

    Returns:
        np.ndarray: The mutated offspring.
    """
    mask_mut = np.random.choice([True, False], size=genes.shape, 
                                p=[mutation_probability, 1 - mutation_probability])

    mutations = generate_plausible_mutations(genes, feature_intervals, num_features_mutation_strength)

    offspring = np.where(mask_mut, mutations, genes)

    return offspring

def generate_plausible_mutations(genes: np.ndarray, feature_intervals: list, 
                                 num_features_mutation_strength: float = 0.25) -> np.ndarray:
    """
    Generates plausible mutations for a given set of genes.

    Parameters:
        genes (np.ndarray): The array of genes (population).
        feature_intervals (list): A list of tuples specifying the interval for each feature.
        num_features_mutation_strength (float): The mutation strength for each feature.

    Returns:
        np.ndarray: The mutated genes.
    """
    mutations = np.zeros_like(genes)

    for i in range(genes.shape[1]):
        range_num = feature_intervals[i][1] - feature_intervals[i][0]
        low = -num_features_mutation_strength / 2
        high = num_features_mutation_strength / 2

        mutations[:, i] = range_num * np.random.uniform(low=low, high=high, size=mutations.shape[0])
        mutations[:, i] += genes[:, i]

        # Fix out-of-range values
        mutations[:, i] = np.clip(mutations[:, i], feature_intervals[i][0], feature_intervals[i][1])

    return mutations.astype(int)
