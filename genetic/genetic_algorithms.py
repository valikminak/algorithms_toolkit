import random
from typing import List, Tuple, Callable, Any, Optional, TypeVar, Generic

T = TypeVar('T')  # Type for individuals (solutions)


class GeneticAlgorithm(Generic[T]):
    """
    Generic genetic algorithm framework.

    This class provides a framework for implementing genetic algorithms
    for optimization problems. It handles the main GA loop and requires
    problem-specific functions to be provided.
    """

    def __init__(self,
                 population_size: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elitism_count: int = 2,
                 tournament_size: int = 3,
                 max_generations: int = 100):
        """
        Initialize the genetic algorithm framework.

        Args:
            population_size: Size of the population
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of top individuals to preserve unchanged
            tournament_size: Number of individuals in tournament selection
            max_generations: Maximum number of generations
        """
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.max_generations = max_generations

        # These functions must be set by the user
        self.create_individual: Optional[Callable[[], T]] = None
        self.fitness_function: Optional[Callable[[T], float]] = None
        self.crossover: Optional[Callable[[T, T], Tuple[T, T]]] = None
        self.mutate: Optional[Callable[[T], T]] = None

    def initialize_population(self) -> List[T]:
        """
        Initialize a random population.

        Returns:
            List of random individuals
        """
        if self.create_individual is None:
            raise ValueError("create_individual function must be set")

        return [self.create_individual() for _ in range(self.population_size)]

    def evaluate_population(self, population: List[T]) -> List[Tuple[T, float]]:
        """
        Evaluate fitness for the entire population.

        Args:
            population: List of individuals

        Returns:
            List of (individual, fitness) tuples
        """
        if self.fitness_function is None:
            raise ValueError("fitness_function must be set")

        return [(individual, self.fitness_function(individual)) for individual in population]

    def tournament_selection(self, evaluated_population: List[Tuple[T, float]]) -> T:
        """
        Select an individual using tournament selection.

        Args:
            evaluated_population: List of (individual, fitness) tuples

        Returns:
            Selected individual
        """
        # Select random individuals for the tournament
        tournament = random.sample(evaluated_population, min(self.tournament_size, len(evaluated_population)))

        # Return the best individual in the tournament
        return max(tournament, key=lambda x: x[1])[0]

    def create_next_generation(self, evaluated_population: List[Tuple[T, float]]) -> List[T]:
        """
        Create the next generation using selection, crossover, and mutation.

        Args:
            evaluated_population: List of (individual, fitness) tuples

        Returns:
            Next generation of individuals
        """
        if self.crossover is None or self.mutate is None:
            raise ValueError("crossover and mutate functions must be set")

        # Sort population by fitness (assuming higher is better)
        sorted_population = sorted(evaluated_population, key=lambda x: x[1], reverse=True)

        # Apply elitism - keep the best individuals
        next_generation = [ind for ind, _ in sorted_population[:self.elitism_count]]

        # Fill the rest of the population with children
        while len(next_generation) < self.population_size:
            # Select parents
            parent1 = self.tournament_selection(evaluated_population)
            parent2 = self.tournament_selection(evaluated_population)

            # Apply crossover with certain probability
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            # Apply mutation with certain probability
            if random.random() < self.mutation_rate:
                child1 = self.mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self.mutate(child2)

            # Add children to next generation
            next_generation.append(child1)
            if len(next_generation) < self.population_size:
                next_generation.append(child2)

        return next_generation

    def run(self, verbose: bool = True) -> Tuple[T, float, List[float]]:
        """
        Run the genetic algorithm.

        Args:
            verbose: Whether to print progress information

        Returns:
            Tuple of (best_individual, best_fitness, fitness_history)
        """
        # Initialize population
        population = self.initialize_population()

        # Track best solution and fitness history
        best_individual = None
        best_fitness = float('-inf')
        fitness_history = []

        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate population
            evaluated_population = self.evaluate_population(population)

            # Get best individual and average fitness
            current_best = max(evaluated_population, key=lambda x: x[1])
            avg_fitness = sum(fitness for _, fitness in evaluated_population) / len(evaluated_population)

            # Update best individual if better
            if current_best[1] > best_fitness:
                best_individual, best_fitness = current_best

            # Store fitness history
            fitness_history.append(avg_fitness)

            # Print progress
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness}, Average Fitness = {avg_fitness}")

            # Create next generation
            population = self.create_next_generation(evaluated_population)

        if verbose:
            print(f"Final result: Best Fitness = {best_fitness}")

        return best_individual, best_fitness, fitness_history


class TSPGeneticSolver:
    """
    Genetic Algorithm solver for the Traveling Salesman Problem (TSP).

    This class uses a genetic algorithm to find an approximate solution
    to the Traveling Salesman Problem.
    """

    def __init__(self,
                 distance_matrix: List[List[float]],
                 population_size: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elitism_count: int = 2,
                 max_generations: int = 100):
        """
        Initialize the TSP genetic solver.

        Args:
            distance_matrix: Matrix of distances between cities
            population_size: Size of the population
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of top individuals to preserve unchanged
            max_generations: Maximum number of generations
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)

        # Initialize genetic algorithm
        self.ga = GeneticAlgorithm[List[int]](
            population_size=population_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_count=elitism_count,
            max_generations=max_generations
        )

        # Set problem-specific functions
        self.ga.create_individual = self._create_individual
        self.ga.fitness_function = self._calculate_fitness
        self.ga.crossover = self._ordered_crossover
        self.ga.mutate = self._swap_mutation

    def _create_individual(self) -> List[int]:
        """
        Create a random tour (individual).

        Returns:
            Random permutation of cities (0 to num_cities-1)
        """
        individual = list(range(self.num_cities))
        random.shuffle(individual)
        return individual

    def _calculate_fitness(self, individual: List[int]) -> float:
        """
        Calculate fitness of an individual (tour).

        Args:
            individual: Tour represented as a list of city indices

        Returns:
            Fitness value (reciprocal of tour length)
        """
        total_distance = 0
        for i in range(self.num_cities):
            from_city = individual[i]
            to_city = individual[(i + 1) % self.num_cities]
            total_distance += self.distance_matrix[from_city][to_city]

        # Return reciprocal (because we want to maximize fitness)
        return 1 / total_distance if total_distance > 0 else 0

    def _ordered_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Perform ordered crossover (OX) between two parents.

        Args:
            parent1, parent2: Parent tours

        Returns:
            Two offspring tours
        """
        size = self.num_cities

        # Select crossover points
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)

        # Create children with placeholder values
        child1 = [-1] * size
        child2 = [-1] * size

        # Copy segment from parents
        for i in range(start, end + 1):
            child1[i] = parent1[i]
            child2[i] = parent2[i]

        # Fill remaining positions with values from other parent
        self._fill_remaining(parent2, child1, start, end)
        self._fill_remaining(parent1, child2, start, end)

        return child1, child2

    def _fill_remaining(self, parent: List[int], child: List[int], start: int, end: int) -> None:
        """
        Fill remaining positions in a child from a parent.

        Args:
            parent: Parent tour
            child: Child tour (with some positions filled)
            start, end: Positions already filled in child
        """
        size = self.num_cities

        # Create a list of cities in the order they appear in the parent
        # but skip those that are already in the child
        ordered_cities = [city for city in parent if city not in child[start:end + 1]]

        # Fill in the remaining positions
        index = 0
        for i in range(size):
            if child[i] == -1:  # If position is empty
                child[i] = ordered_cities[index]
                index += 1

    def _swap_mutation(self, individual: List[int]) -> List[int]:
        """
        Perform swap mutation on an individual.

        Args:
            individual: Tour to mutate

        Returns:
            Mutated tour
        """
        # Create a copy of the individual
        mutated = individual.copy()

        # Select two random positions and swap them
        pos1 = random.randint(0, self.num_cities - 1)
        pos2 = random.randint(0, self.num_cities - 1)

        while pos1 == pos2:
            pos2 = random.randint(0, self.num_cities - 1)

        mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]

        return mutated

    def solve(self, verbose: bool = True) -> Tuple[List[int], float]:
        """
        Solve the TSP using genetic algorithm.

        Args:
            verbose: Whether to print progress information

        Returns:
            Tuple of (best_tour, tour_length)
        """
        # Run the genetic algorithm
        best_tour, best_fitness, _ = self.ga.run(verbose=verbose)

        # Calculate the tour length
        tour_length = 1 / best_fitness

        return best_tour, tour_length


class BinaryGeneticAlgorithm:
    """
    Genetic Algorithm for optimization with binary encoding.

    This class implements a genetic algorithm where individuals are
    represented as binary strings (lists of 0s and 1s).
    """

    def __init__(self,
                 chromosome_length: int,
                 fitness_function: Callable[[List[int]], float],
                 population_size: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.01,
                 elitism_count: int = 2,
                 max_generations: int = 100):
        """
        Initialize the binary genetic algorithm.

        Args:
            chromosome_length: Length of binary chromosomes
            fitness_function: Function to evaluate fitness of a chromosome
            population_size: Size of the population
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation per bit
            elitism_count: Number of top individuals to preserve unchanged
            max_generations: Maximum number of generations
        """
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.max_generations = max_generations

    def initialize_population(self) -> List[List[int]]:
        """
        Initialize a random population of binary chromosomes.

        Returns:
            List of random binary chromosomes
        """
        return [
            [random.randint(0, 1) for _ in range(self.chromosome_length)]
            for _ in range(self.population_size)
        ]

    def evaluate_population(self, population: List[List[int]]) -> List[Tuple[List[int], float]]:
        """
        Evaluate fitness for the entire population.

        Args:
            population: List of chromosomes

        Returns:
            List of (chromosome, fitness) tuples
        """
        return [(chrom, self.fitness_function(chrom)) for chrom in population]

    def select_parents(self, evaluated_population: List[Tuple[List[int], float]]) -> List[List[int]]:
        """
        Select parents using roulette wheel selection.

        Args:
            evaluated_population: List of (chromosome, fitness) tuples

        Returns:
            List of selected parent chromosomes
        """
        total_fitness = sum(fitness for _, fitness in evaluated_population)

        # If total fitness is zero or negative, use uniform selection
        if total_fitness <= 0:
            return [chrom for chrom, _ in random.sample(evaluated_population, self.population_size)]

        # Otherwise, use roulette wheel selection
        selected = []
        for _ in range(self.population_size):
            # Spin the wheel
            pick = random.uniform(0, total_fitness)
            current = 0

            for chromosome, fitness in evaluated_population:
                current += fitness
                if current > pick:
                    selected.append(chromosome)
                    break

            # If we didn't select anyone (due to floating point issues), add last individual
            if len(selected) <= _ and evaluated_population:
                selected.append(evaluated_population[-1][0])

        return selected

    def crossover(self, parents: List[List[int]]) -> List[List[int]]:
        """
        Perform crossover on selected parents.

        Args:
            parents: List of parent chromosomes

        Returns:
            List of offspring chromosomes
        """
        offspring = []

        for i in range(0, len(parents), 2):
            # If there's an odd number of parents, copy the last one directly
            if i == len(parents) - 1:
                offspring.append(parents[i].copy())
                continue

            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Apply crossover with certain probability
            if random.random() < self.crossover_rate:
                # Single-point crossover
                crossover_point = random.randint(1, self.chromosome_length - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                # No crossover, just copy the parents
                child1 = parent1.copy()
                child2 = parent2.copy()

            offspring.append(child1)
            offspring.append(child2)

        return offspring

    def mutate(self, chromosomes: List[List[int]]) -> List[List[int]]:
        """
        Apply mutation to chromosomes.

        Args:
            chromosomes: List of chromosomes to mutate

        Returns:
            List of mutated chromosomes
        """
        mutated = []

        for chrom in chromosomes:
            new_chrom = chrom.copy()

            # Apply mutation with certain probability for each bit
            for i in range(self.chromosome_length):
                if random.random() < self.mutation_rate:
                    new_chrom[i] = 1 - new_chrom[i]  # Flip the bit

            mutated.append(new_chrom)

        return mutated

    def create_next_generation(self, evaluated_population: List[Tuple[List[int], float]]) -> List[List[int]]:
        """
        Create the next generation using selection, crossover, and mutation.

        Args:
            evaluated_population: List of (chromosome, fitness) tuples

        Returns:
            Next generation of chromosomes
        """
        # Sort population by fitness (assuming higher is better)
        sorted_population = sorted(evaluated_population, key=lambda x: x[1], reverse=True)

        # Apply elitism - keep the best chromosomes
        elites = [chrom for chrom, _ in sorted_population[:self.elitism_count]]

        # Select parents
        parents = self.select_parents(evaluated_population)

        # Apply crossover
        offspring = self.crossover(parents)

        # Apply mutation
        mutated_offspring = self.mutate(offspring)

        # Create new generation with elites and mutated offspring
        next_generation = elites + mutated_offspring[:self.population_size - len(elites)]

        return next_generation

    def run(self, verbose: bool = True) -> Tuple[List[int], float, List[float]]:
        """
        Run the genetic algorithm.

        Args:
            verbose: Whether to print progress information

        Returns:
            Tuple of (best_chromosome, best_fitness, fitness_history)
        """
        # Initialize population
        population = self.initialize_population()

        # Track best solution and fitness history
        best_chromosome = None
        best_fitness = float('-inf')
        fitness_history = []

        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate population
            evaluated_population = self.evaluate_population(population)

            # Get best individual and average fitness
            current_best = max(evaluated_population, key=lambda x: x[1])
            avg_fitness = sum(fitness for _, fitness in evaluated_population) / len(evaluated_population)

            # Update best individual if better
            if current_best[1] > best_fitness:
                best_chromosome, best_fitness = current_best

            # Store fitness history
            fitness_history.append(avg_fitness)

            # Print progress
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness}, Average Fitness = {avg_fitness}")

            # Create next generation
            population = self.create_next_generation(evaluated_population)

        if verbose:
            print(f"Final result: Best Fitness = {best_fitness}")

        return best_chromosome, best_fitness, fitness_history


class NSGA2:
    """
    Non-dominated Sorting Genetic Algorithm II (NSGA-II) for multi-objective optimization.

    NSGA-II is an extension of the genetic algorithm for solving multi-objective
    optimization problems, producing a set of Pareto-optimal solutions.
    """

    def __init__(self,
                 num_objectives: int,
                 chromosome_length: int,
                 objective_functions: List[Callable[[List[Any]], float]],
                 create_chromosome: Callable[[], List[Any]],
                 crossover_function: Callable[[List[Any], List[Any]], Tuple[List[Any], List[Any]]],
                 mutation_function: Callable[[List[Any]], List[Any]],
                 population_size: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 max_generations: int = 100):
        """
        Initialize the NSGA-II algorithm.

        Args:
            num_objectives: Number of objective functions
            chromosome_length: Length of chromosomes
            objective_functions: List of objective functions to minimize
            create_chromosome: Function to create a random chromosome
            crossover_function: Function to perform crossover on two parents
            mutation_function: Function to mutate a chromosome
            population_size: Size of the population
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            max_generations: Maximum number of generations
        """
        self.num_objectives = num_objectives
        self.chromosome_length = chromosome_length
        self.objective_functions = objective_functions
        self.create_chromosome = create_chromosome
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations

    def initialize_population(self) -> List[List[Any]]:
        """
        Initialize a random population.

        Returns:
            List of random chromosomes
        """
        return [self.create_chromosome() for _ in range(self.population_size)]

    def evaluate_population(self, population: List[List[Any]]) -> List[Tuple[List[Any], List[float]]]:
        """
        Evaluate all objective functions for the population.

        Args:
            population: List of chromosomes

        Returns:
            List of (chromosome, objective_values) tuples
        """
        evaluated = []

        for chrom in population:
            objective_values = [obj_func(chrom) for obj_func in self.objective_functions]
            evaluated.append((chrom, objective_values))

        return evaluated

    def dominates(self, obj_values1: List[float], obj_values2: List[float]) -> bool:
        """
        Check if solution 1 dominates solution 2.

        A solution dominates another if it is not worse in any objective
        and strictly better in at least one objective.

        Args:
            obj_values1: Objective values of solution 1
            obj_values2: Objective values of solution 2

        Returns:
            True if solution 1 dominates solution 2, False otherwise
        """
        not_worse = all(o1 <= o2 for o1, o2 in zip(obj_values1, obj_values2))
        strictly_better = any(o1 < o2 for o1, o2 in zip(obj_values1, obj_values2))
        return not_worse and strictly_better

    def fast_non_dominated_sort(self, evaluated_population: List[Tuple[List[Any], List[float]]]
                                ) -> List[List[int]]:
        """
        Sort the population into non-dominated fronts.

        Args:
            evaluated_population: List of (chromosome, objective_values) tuples

        Returns:
            List of fronts, where each front is a list of indices into evaluated_population
        """
        n = len(evaluated_population)
        domination_count = [0] * n  # Number of solutions that dominate solution i
        dominated_solutions = [[] for _ in range(n)]  # List of solutions that solution i dominates
        fronts = [[]]  # First front

        # For each solution
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                obj_values_i = evaluated_population[i][1]
                obj_values_j = evaluated_population[j][1]

                # If i dominates j
                if self.dominates(obj_values_i, obj_values_j):
                    dominated_solutions[i].append(j)
                # If j dominates i
                elif self.dominates(obj_values_j, obj_values_i):
                    domination_count[i] += 1

            # If solution i belongs to the first front
            if domination_count[i] == 0:
                fronts[0].append(i)

        # Find remaining fronts
        front_index = 0
        while fronts[front_index]:
            next_front = []

            for i in fronts[front_index]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1

                    if domination_count[j] == 0:
                        next_front.append(j)

            front_index += 1
            fronts.append(next_front)

        # Remove the empty front at the end
        fronts.pop()

        return fronts

    def crowding_distance_assignment(self, evaluated_population: List[Tuple[List[Any], List[float]]],
                                     front: List[int]) -> List[float]:
        """
        Calculate crowding distance for solutions in a front.

        Args:
            evaluated_population: List of (chromosome, objective_values) tuples
            front: List of indices into evaluated_population

        Returns:
            List of crowding distances for each solution in the front
        """
        if not front:
            return []

        front_size = len(front)
        distances = [0.0] * front_size

        # For each objective
        for obj_index in range(self.num_objectives):
            # Sort front by the objective
            sorted_front = sorted(
                range(front_size),
                key=lambda i: evaluated_population[front[i]][1][obj_index]
            )

            # Give infinite distance to boundary points
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')

            # Calculate distances for intermediate points
            if front_size > 2:
                obj_range = (
                        evaluated_population[front[sorted_front[-1]]][1][obj_index] -
                        evaluated_population[front[sorted_front[0]]][1][obj_index]
                )

                if obj_range == 0:
                    continue

                for i in range(1, front_size - 1):
                    distances[sorted_front[i]] += (
                                                          evaluated_population[front[sorted_front[i + 1]]][1][
                                                              obj_index] -
                                                          evaluated_population[front[sorted_front[i - 1]]][1][obj_index]
                                                  ) / obj_range

        return distances

    def tournament_selection(self, evaluated_population: List[Tuple[List[Any], List[float]]],
                             fronts: List[List[int]], crowding_distances: List[List[float]]
                             ) -> List[List[Any]]:
        """
        Select parents using binary tournament selection based on rank and crowding distance.

        Args:
            evaluated_population: List of (chromosome, objective_values) tuples
            fronts: List of fronts from non-dominated sorting
            crowding_distances: List of crowding distances for each front

        Returns:
            List of selected parent chromosomes
        """
        parents = []

        # Convert fronts and crowding distances to a more convenient form
        rank = {}
        distance = {}

        for i, front in enumerate(fronts):
            for j, solution_idx in enumerate(front):
                rank[solution_idx] = i
                distance[solution_idx] = crowding_distances[i][j]

        # Select parents using binary tournament
        for _ in range(self.population_size):
            # Select two random individuals
            a, b = random.sample(range(len(evaluated_population)), 2)

            # Select the better one based on rank and crowding distance
            if rank[a] < rank[b] or (rank[a] == rank[b] and distance[a] > distance[b]):
                parents.append(evaluated_population[a][0])
            else:
                parents.append(evaluated_population[b][0])

        return parents

    def create_offspring(self, parents: List[List[Any]]) -> List[List[Any]]:
        """
        Create offspring through crossover and mutation.

        Args:
            parents: List of parent chromosomes

        Returns:
            List of offspring chromosomes
        """
        offspring = []

        for i in range(0, len(parents), 2):
            # If there's an odd number of parents, handle the last one
            if i == len(parents) - 1:
                child = parents[i].copy()

                # Apply mutation with certain probability
                if random.random() < self.mutation_rate:
                    child = self.mutation_function(child)

                offspring.append(child)
                continue

            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Apply crossover with certain probability
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover_function(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Apply mutation with certain probability
            if random.random() < self.mutation_rate:
                child1 = self.mutation_function(child1)
            if random.random() < self.mutation_rate:
                child2 = self.mutation_function(child2)

            offspring.append(child1)
            offspring.append(child2)

        return offspring[:self.population_size]

    def run(self, verbose: bool = True) -> List[Tuple[List[Any], List[float]]]:
        """
        Run the NSGA-II algorithm.

        Args:
            verbose: Whether to print progress information

        Returns:
            Pareto-optimal solutions as (chromosome, objective_values) tuples
        """
        # Initialize population
        population = self.initialize_population()

        # Evaluate initial population
        evaluated_population = self.evaluate_population(population)

        # Evolution loop
        for generation in range(self.max_generations):
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(evaluated_population)

            # Calculate crowding distance for each front
            crowding_distances = [
                self.crowding_distance_assignment(evaluated_population, front)
                for front in fronts
            ]

            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Number of fronts = {len(fronts)}")
                print(f"First front size = {len(fronts[0])}")

            # Select parents
            parents = self.tournament_selection(evaluated_population, fronts, crowding_distances)

            # Create offspring
            offspring = self.create_offspring(parents)

            # Evaluate offspring
            evaluated_offspring = self.evaluate_population(offspring)

            # Combine parent and offspring populations
            combined_population = evaluated_population + evaluated_offspring

            # Non-dominated sorting of combined population
            combined_fronts = self.fast_non_dominated_sort(combined_population)

            # Select the next generation population
            new_population = []
            new_evaluated = []

            for front in combined_fronts:
                if len(new_population) + len(front) <= self.population_size:
                    # Add all solutions in this front
                    for idx in front:
                        new_population.append(combined_population[idx][0])
                        new_evaluated.append(combined_population[idx])
                else:
                    # Calculate crowding distance
                    front_distances = self.crowding_distance_assignment(combined_population, front)

                    # Sort front by crowding distance (descending)
                    sorted_front = sorted(
                        range(len(front)),
                        key=lambda i: front_distances[i],
                        reverse=True
                    )

                    # Add solutions until population is full
                    remaining = self.population_size - len(new_population)
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        new_population.append(combined_population[idx][0])
                        new_evaluated.append(combined_population[idx])

                    break

            # Update population
            evaluated_population = new_evaluated

        # Return the Pareto-optimal solutions (first front)
        final_fronts = self.fast_non_dominated_sort(evaluated_population)

        if verbose:
            print(f"Final number of fronts: {len(final_fronts)}")
            print(f"Size of Pareto front: {len(final_fronts[0])}")

        return [evaluated_population[i] for i in final_fronts[0]]

    class DifferentialEvolution:
        """
        Differential Evolution (DE) optimization algorithm.

        DE is a population-based optimization algorithm that uses vector differences
        for mutation operations, and is effective for many optimization problems.
        """

        def __init__(self,
                     fitness_function: Callable[[List[float]], float],
                     bounds: List[Tuple[float, float]],
                     population_size: int = 50,
                     f: float = 0.5,
                     cr: float = 0.7,
                     max_generations: int = 100,
                     strategy: str = 'rand/1/bin'):
            """
            Initialize the Differential Evolution algorithm.

            Args:
                fitness_function: Function to optimize (minimize)
                bounds: List of (min, max) bounds for each variable
                population_size: Size of the population
                f: Scaling factor (typically in [0.5, 1.0])
                cr: Crossover probability (typically in [0.7, 1.0])
                max_generations: Maximum number of generations
                strategy: DE strategy ('rand/1/bin', 'best/1/bin', 'rand/2/bin', etc.)
            """
            self.fitness_function = fitness_function
            self.bounds = bounds
            self.dim = len(bounds)
            self.population_size = population_size
            self.f = f
            self.cr = cr
            self.max_generations = max_generations
            self.strategy = strategy

            # Parse strategy
            self._parse_strategy()

        def _parse_strategy(self):
            """Parse the DE strategy string."""
            parts = self.strategy.split('/')
            if len(parts) != 3:
                raise ValueError(f"Invalid strategy: {self.strategy}")

            self.base = parts[0]  # 'rand' or 'best'
            self.n_vectors = int(parts[1])  # Number of difference vectors
            self.crossover_type = parts[2]  # 'bin' (binomial) or 'exp' (exponential)

        def initialize_population(self) -> List[List[float]]:
            """
            Initialize a random population within the bounds.

            Returns:
                List of random solution vectors
            """
            population = []

            for _ in range(self.population_size):
                # Create a random vector within bounds
                vector = [random.uniform(low, high) for low, high in self.bounds]
                population.append(vector)

            return population

        def evaluate_population(self, population: List[List[float]]) -> List[float]:
            """
            Evaluate fitness for the entire population.

            Args:
                population: List of solution vectors

            Returns:
                List of fitness values
            """
            return [self.fitness_function(ind) for ind in population]

        def mutation(self, population: List[List[float]], fitness_values: List[float]) -> List[List[float]]:
            """
            Create mutant vectors according to the DE strategy.

            Args:
                population: Current population
                fitness_values: Fitness values of the population

            Returns:
                List of mutant vectors
            """
            mutants = []

            for i in range(self.population_size):
                # Select base vector
                if self.base == 'rand':
                    # Random individual from the population
                    base_idx = random.randint(0, self.population_size - 1)
                    base = population[base_idx]
                elif self.base == 'best':
                    # Best individual in the population
                    best_idx = fitness_values.index(min(fitness_values))
                    base = population[best_idx]
                else:
                    raise ValueError(f"Unknown base selection: {self.base}")

                # Create difference vectors
                diff_vectors = []
                for _ in range(self.n_vectors):
                    # Select two random individuals different from base
                    r1, r2 = random.sample(range(self.population_size), 2)
                    while r1 == base_idx or r2 == base_idx:
                        r1, r2 = random.sample(range(self.population_size), 2)

                    # Calculate difference vector
                    diff = [population[r1][j] - population[r2][j] for j in range(self.dim)]
                    diff_vectors.append(diff)

                # Calculate mutant vector
                mutant = base.copy()
                for diff in diff_vectors:
                    for j in range(self.dim):
                        mutant[j] += self.f * diff[j]

                # Ensure mutant is within bounds
                for j in range(self.dim):
                    if mutant[j] < self.bounds[j][0]:
                        mutant[j] = self.bounds[j][0]
                    elif mutant[j] > self.bounds[j][1]:
                        mutant[j] = self.bounds[j][1]

                mutants.append(mutant)

            return mutants

        def crossover(self, population: List[List[float]], mutants: List[List[float]]) -> List[List[float]]:
            """
            Create trial vectors through crossover.

            Args:
                population: Current population
                mutants: Mutant vectors

            Returns:
                List of trial vectors
            """
            trials = []

            for i in range(self.population_size):
                trial = []

                # Ensure at least one parameter is changed
                j_rand = random.randint(0, self.dim - 1)

                if self.crossover_type == 'bin':
                    # Binomial crossover
                    for j in range(self.dim):
                        if random.random() < self.cr or j == j_rand:
                            trial.append(mutants[i][j])
                        else:
                            trial.append(population[i][j])

                elif self.crossover_type == 'exp':
                    # Exponential crossover
                    trial = population[i].copy()
                    j = j_rand

                    # Copy parameters from mutant until CR test fails
                    trial[j] = mutants[i][j]
                    j = (j + 1) % self.dim

                    while random.random() < self.cr and j != j_rand:
                        trial[j] = mutants[i][j]
                        j = (j + 1) % self.dim

                else:
                    raise ValueError(f"Unknown crossover type: {self.crossover_type}")

                trials.append(trial)

            return trials

        def selection(self, population: List[List[float]], trials: List[List[float]],
                      fitness_values: List[float]) -> Tuple[List[List[float]], List[float]]:
            """
            Select the better vectors between the current population and trial vectors.

            Args:
                population: Current population
                trials: Trial vectors
                fitness_values: Fitness values of the current population

            Returns:
                Tuple of (new_population, new_fitness_values)
            """
            new_population = []
            new_fitness_values = []

            for i in range(self.population_size):
                # Evaluate the trial vector
                trial_fitness = self.fitness_function(trials[i])

                # Select the better vector
                if trial_fitness <= fitness_values[i]:
                    new_population.append(trials[i])
                    new_fitness_values.append(trial_fitness)
                else:
                    new_population.append(population[i])
                    new_fitness_values.append(fitness_values[i])

            return new_population, new_fitness_values

        def run(self, verbose: bool = True) -> Tuple[List[float], float, List[float]]:
            """
            Run the Differential Evolution algorithm.

            Args:
                verbose: Whether to print progress information

            Returns:
                Tuple of (best_solution, best_fitness, fitness_history)
            """
            # Initialize population
            population = self.initialize_population()

            # Evaluate initial population
            fitness_values = self.evaluate_population(population)

            # Track the best solution and fitness history
            best_idx = fitness_values.index(min(fitness_values))
            best_solution = population[best_idx].copy()
            best_fitness = fitness_values[best_idx]
            fitness_history = [best_fitness]

            # Evolution loop
            for generation in range(self.max_generations):
                # Create mutant vectors
                mutants = self.mutation(population, fitness_values)

                # Create trial vectors
                trials = self.crossover(population, mutants)

                # Selection
                population, fitness_values = self.selection(population, trials, fitness_values)

                # Update best solution
                current_best = min(fitness_values)
                if current_best < best_fitness:
                    best_idx = fitness_values.index(current_best)
                    best_solution = population[best_idx].copy()
                    best_fitness = current_best

                # Store best fitness in history
                fitness_history.append(best_fitness)

                # Print progress
                if verbose and generation % 10 == 0:
                    print(f"Generation {generation}: Best Fitness = {best_fitness}")

            if verbose:
                print(f"Final result: Best Fitness = {best_fitness}")

            return best_solution, best_fitness, fitness_history