import time
import copy
import numpy as np
from PIL import Image
from multiprocess import Pool, cpu_count
from vangogh import selection, variation
from vangogh.fitness import drawing_fitness_function, draw_voronoi_image
from vangogh.population import Population
from vangogh.util import NUM_VARIABLES_PER_POINT, IMAGE_SHRINK_SCALE, REFERENCE_IMAGE
from vangogh.rbf import RBFN


class Evolution:
    """
    Class to handle the evolutionary process for generating images using a Voronoi-based genetic algorithm.
    """

    def __init__(self, num_points: int, reference_image: Image, evolution_type: str = 'p+o',
                 population_size: int = 200, generation_budget: int = -1, evaluation_budget: int = -1,
                 crossover_method: str = "ONE_POINT", mutation_probability: str = 'inv_mutable_genotype_length',
                 num_features_mutation_strength: float = 0.25, num_features_mutation_strength_decay: float = None,
                 num_features_mutation_strength_decay_generations: list = None, selection_name: str = 'tournament_2',
                 initialization: str = 'RANDOM', noisy_evaluations: bool = False, verbose: bool = False,
                 ref_image_name: str = "wheat_field", generation_reporter=None, seed: int = 0):
        """
        Initializes the Evolution object with the specified parameters.
        """
        self.curr_gen = 0
        self.selection_method = selection_name

        # Prepare the reference image
        self.reference_image = self._prepare_reference_image(reference_image)

        # Define feature intervals for points
        self.feature_intervals = self._define_feature_intervals(num_points)

        # Store parameters
        self.num_points = num_points
        self.evolution_type = evolution_type
        self.population_size = population_size
        self.generation_budget = generation_budget
        self.evaluation_budget = evaluation_budget
        self.mutation_probability = mutation_probability
        self.num_features_mutation_strength = num_features_mutation_strength
        self.num_features_mutation_strength_decay = num_features_mutation_strength_decay
        self.num_features_mutation_strength_decay_generations = num_features_mutation_strength_decay_generations
        self.noisy_evaluations = noisy_evaluations
        self.verbose = verbose
        self.generation_reporter = generation_reporter
        self.crossover_method = crossover_method
        self.num_evaluations = 0
        self.initialization = initialization
        self.ref_image_name = ref_image_name

        np.random.seed(seed)
        self.seed = seed

        # Convert feature intervals to a NumPy array
        self.feature_intervals = np.array(self.feature_intervals, dtype=object)

        # Validate tournament selection compatibility
        self._validate_tournament_selection(selection_name)

        # Initialize population and elite
        self.genotype_length = len(self.feature_intervals)
        self.population = Population(self.population_size, self.genotype_length, self.initialization)
        self.elite = None
        self.elite_fitness = np.inf

        # Set mutation probability
        self._set_mutation_probability(mutation_probability)

        # Setup RBFN for crossover
        self.rbfn = RBFN(self.genotype_length, self.genotype_length, self.genotype_length)
        self.rbfn_loss = None
        self.train_set = []
        self._max_size = 100

        # Check for incompatibilities
        self._check_incompatibilities()

    def _prepare_reference_image(self, reference_image: Image) -> Image:
        """
        Prepares and resizes the reference image.

        Args:
            reference_image (Image): The input reference image.

        Returns:
            Image: The resized reference image.
        """
        reference_image_copy = reference_image.copy()
        reference_image_copy.thumbnail(
            (int(reference_image_copy.width / IMAGE_SHRINK_SCALE),
             int(reference_image_copy.height / IMAGE_SHRINK_SCALE)),
            Image.ANTIALIAS
        )
        return reference_image_copy

    def _define_feature_intervals(self, num_points: int) -> list:
        """
        Defines the intervals for the features (points and color).

        Args:
            num_points (int): Number of points.

        Returns:
            list: Intervals for each feature.
        """
        num_variables = num_points * NUM_VARIABLES_PER_POINT
        return [
            [0, self.reference_image.width] if i % NUM_VARIABLES_PER_POINT == 0 else
            [0, self.reference_image.height] if i % NUM_VARIABLES_PER_POINT == 1 else
            [0, 256] for i in range(num_variables)
        ]

    def _validate_tournament_selection(self, selection_name: str):
        """
        Validates that the tournament selection is compatible with the population size.

        Args:
            selection_name (str): Name of the selection method.

        Raises:
            ValueError: If the population size is not a multiple of the tournament size.
        """
        if 'tournament' in selection_name:
            self.tournament_size = int(selection_name.split('_')[-1])
            if self.population_size % self.tournament_size != 0:
                raise ValueError('Population size must be a multiple of the tournament size')

    def _set_mutation_probability(self, mutation_probability: str):
        """
        Sets the mutation probability based on the provided method.

        Args:
            mutation_probability (str): The method to determine mutation probability.
        """
        if mutation_probability == 'inv_genotype_length':
            self.mutation_probability = 1 / self.genotype_length
        elif mutation_probability == "inv_mutable_genotype_length":
            num_unmutable_features = 0
            self.mutation_probability = 1 / (self.genotype_length - num_unmutable_features)

    def _check_incompatibilities(self):
        """
        Checks for incompatibilities in the evolution settings and raises errors/warnings if found.
        """
        if self.evolution_type == 'p+o' and self.noisy_evaluations:
            raise ValueError("P+O is not compatible with noisy evaluations.")
        elif 'age_reg' in self.evolution_type:
            print("Warning: noisy evaluations and age-regularized evolution may not be fully compatible.")

    def __update_elite(self, population: Population):
        """
        Updates the elite individual in the population.

        Args:
            population (Population): The current population.
        """
        best_fitness_idx = np.argmin(population.fitnesses)
        best_fitness = population.fitnesses[best_fitness_idx]
        if self.noisy_evaluations or best_fitness < self.elite_fitness:
            self.elite = population.genes[best_fitness_idx, :].copy()
            self.elite_fitness = best_fitness

    def __classic_generation(self, merge_parent_offspring: bool = False):
        """
        Performs a classic generation step, including selection, crossover, mutation, and updating the population.

        Args:
            merge_parent_offspring (bool): Whether to merge parent and offspring populations.
        """
        offspring = Population(self.population_size, self.genotype_length, self.initialization)
        offspring.genes[:] = self.population.genes[:]
        offspring.shuffle()

        parents_fitnesses = drawing_fitness_function(self.population.genes, self.reference_image)
        parent_genes = copy.deepcopy(offspring.genes)

        offspring.genes, _ = variation.crossover(offspring.genes, self.crossover_method, rbfn=self.rbfn)
        offspring.genes = variation.mutate(
            offspring.genes, self.feature_intervals, mutation_probability=self.mutation_probability,
            num_features_mutation_strength=self.num_features_mutation_strength
        )

        offspring.fitnesses = drawing_fitness_function(offspring.genes, self.reference_image)
        self.num_evaluations += len(offspring.genes)

        self.__update_elite(offspring)

        # Train the RBF network to improve crossover
        if self.crossover_method == "RBFNX":
            self._train_rbfn(parent_genes, parents_fitnesses)

        # Perform selection and population update
        if merge_parent_offspring:
            self.population.stack(offspring)
        else:
            self.population = offspring

        self.population = selection.select(
            self.population, self.population_size, self.curr_gen, selection_name=self.selection_name
        )

    def _train_rbfn(self, parent_genes, parents_fitnesses):
        """
        Trains the RBF network for improved crossover.

        Args:
            parent_genes (array): Parent genes before crossover.
            parents_fitnesses (array): Fitnesses of the parent genes.
        """
        uniformx_genes, uniformx_crossover = variation.crossover(parent_genes, "UNIFORM", rbfn=self.rbfn)
        uniformx_fitness = drawing_fitness_function(uniformx_genes, self.reference_image)

        parents_a = np.array(parent_genes[:len(parent_genes) // 2])
        parents_b = np.array(parent_genes[len(parent_genes) // 2:])

        parents_a_fitnesses = np.array(parents_fitnesses[:len(parents_fitnesses) // 2])
        parents_b_fitnesses = np.array(parents_fitnesses[len(parents_fitnesses) // 2:])

        for i in range(len(parents_a)):
            if uniformx_fitness[i] < parents_a_fitnesses[i] and uniformx_fitness[i] < parents_b_fitnesses[i]:
                self.train_set.append([parents_a[i], parents_b[i], uniformx_crossover[i]])

        if len(self.train_set) % self._max_size == 0:
            parents_a, parents_b, uniformx_crossover = zip(*self.train_set)

            parents_a, parents_b = np.array(parents_a), np.array(parents_b)
            uniformx_crossover = np.array(uniformx_crossover)

            self.rbfn_loss = self.rbfn.train_rbf(parents_a, parents_b, uniformx_crossover)

    def run(self):
        """
        Runs the evolutionary process for the specified number of generations or evaluations.

        Returns:
            list: Data collected during the run, including generation number, evaluations, and fitness.
        """
        data = []
        self._initialize_population()

        start_time_seconds = time.time()

        # Main evolution loop
        while not self._evolution_terminated():
            self._decay_mutation_strength()
            self._perform_evolutionary_step()

            if self.verbose:
                self._print_generation_summary()

            data.append(self._collect_generation_data(start_time_seconds))

            if self.generation_reporter is not None:
                self._report_generation(start_time_seconds)

        self._save_final_image()
        return data

    def _initialize_population(self):
        """
        Initializes the population and computes initial fitnesses.
        """
        self.population.initialize(self.feature_intervals)
        self.population.fitnesses = drawing_fitness_function(self.population.genes, self.reference_image)
        self.num_evaluations = len(self.population.genes)
        self.__update_elite(self.population)

    def _decay_mutation_strength(self):
        """
        Decays the mutation strength if applicable based on the generation number.
        """
        if self.num_features_mutation_strength_decay_generations is not None:
            if self.curr_gen in self.num_features_mutation_strength_decay_generations:
                self.num_features_mutation_strength *= self.num_features_mutation_strength_decay

    def _perform_evolutionary_step(self):
        """
        Performs a single step of evolution based on the specified evolution type.
        """
        if self.evolution_type == 'classic':
            self.__classic_generation(merge_parent_offspring=False)
        elif self.evolution_type == 'p+o':
            self.__classic_generation(merge_parent_offspring=True)
        else:
            raise ValueError(f'Unknown evolution type: {self.evolution_type}')

        self.curr_gen += 1

    def _evolution_terminated(self) -> bool:
        """
        Determines if the evolution process should terminate based on the generation or evaluation budget, 
        or if the population has converged.

        Returns:
            bool: True if the evolution should terminate, False otherwise.
        """
        if 0 < self.generation_budget <= self.curr_gen:
            return True
        if 0 < self.evaluation_budget <= self.num_evaluations:
            return True
        if self.population.is_converged():
            return True
        return False

    def _collect_generation_data(self, start_time_seconds: float) -> dict:
        """
        Collects data for the current generation.

        Args:
            start_time_seconds (float): The start time of the evolution process.

        Returns:
            dict: Collected data for the current generation.
        """
        return {
            "num-generations": self.curr_gen,
            "num-evaluations": self.num_evaluations,
            "time-elapsed": time.time() - start_time_seconds,
            "best-fitness": self.elite_fitness,
            "crossover-method": self.crossover_method,
            "population-size": self.population_size,
            "num-points": self.num_points,
            "initialization": self.initialization,
            "seed": self.seed,
            "rbfn-loss": self.rbfn_loss,
            "selection-method": self.selection_method,
            "ref_image_name": self.ref_image_name
        }

    def _print_generation_summary(self):
        """
        Prints a summary of the current generation if verbose mode is enabled.
        """
        print(f'Generation: {self.curr_gen}, Best Fitness: {self.elite_fitness}, '
              f'Avg. Fitness: {np.mean(self.population.fitnesses)}')

    def _report_generation(self, start_time_seconds: float):
        """
        Reports the current generation data using the provided generation reporter.

        Args:
            start_time_seconds (float): The start time of the evolution process.
        """
        self.generation_reporter({
            "num-generations": self.curr_gen,
            "num-evaluations": self.num_evaluations,
            "time-elapsed": time.time() - start_time_seconds
        }, self)

    def _save_final_image(self):
        """
        Saves the final evolved image as a PNG file.
        """
        final_image = draw_voronoi_image(
            self.elite, self.reference_image.width, self.reference_image.height, scale=IMAGE_SHRINK_SCALE
        )
        final_image.save(f"./img/van_gogh_final_{self.seed}_{self.ref_image_name}_{self.population_size}_"
                         f"{self.crossover_method}_{self.num_points}_{self.initialization}_"
                         f"{self.generation_budget}_{self.selection_method}.png")


if __name__ == '__main__':
    evo = Evolution(
        num_points=100, reference_image=REFERENCE_IMAGE, evolution_type='p+o', population_size=100,
        generation_budget=300, crossover_method='ONE_POINT', initialization='RANDOM',
        num_features_mutation_strength=0.25, selection_name='tournament_4', noisy_evaluations=False,
        verbose=True, ref_image_name='wheat_field'
    )
    evo.run()
