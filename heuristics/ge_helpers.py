import copy
import random
import time
from collections import OrderedDict, defaultdict
from numbers import Number
from typing import Any, List, Dict, DefaultDict

from heuristics.grammar import Grammar
from heuristics.ge_graph import PopulatedGraph
from heuristics.population import Individual, Population, CoevPopulation
from fitness.fitness import FitnessFunction, DEFAULT_FITNESS
from util.output_util import (
    print_stats,
    write_run_output,
    write_run_output_coev,
    write_spatial_output,
)

CACHE_MAX_SIZE = 100000


def map_input_with_grammar(individual: Individual, grammar: Grammar) -> Individual:
    """ Generate a sentence from input and set the sentence and number of used
    input.

    :param individual:
    :type individual: Individual
    :param grammar: Grammar used to generate output sentence from input
    :type grammar: Grammar
    :return: individual
    :rtype: Individual

    """
    break_out = 100
    cnt = 0
    phenotype: str = Individual.DEFAULT_PHENOTYPE
    n_inputs_used: int = 0
    while phenotype is Individual.DEFAULT_PHENOTYPE and cnt < break_out:
        phenotype, n_inputs_used = grammar.generate_sentence(individual.genome)
        if phenotype is Individual.DEFAULT_PHENOTYPE:
            _individual = Individual(None)
            individual.genome = _individual.genome
            cnt += 1

    # None phenotype causes stochastic behavior. Can happen since we
    # use a break out counter to avoid infinite loop
    # TODO count number of remappings
    individual.phenotype = phenotype

    # TODO better solution, this handles testing when insensible
    # grammars are passed through. Thus the grammar correctness need
    # to be guaranteed as well...
    if phenotype is Individual.DEFAULT_PHENOTYPE:
        raise ValueError("Phenotype is DEFAULT_PHENOTYPE: {}".format(Individual.DEFAULT_PHENOTYPE))

    individual.used_input = n_inputs_used

    return individual


def initialise_population(size: int) -> List[Individual]:
    """Create a population of Individuals of the given size.

    :param size: Number of individuals to generate
    :type size: int
    :return: Randomly generated individuals
    :rtype: list of Individual
    """
    assert size > 0

    individuals = [Individual(None) for _ in range(size)]

    return individuals


def search_loop(population: Population, param: Dict[str, Any]) -> Individual:
    """Return the best individual from the evolutionary search loop. Assumes
    the population is initially not evaluated.

    :param population: Initial populations for search
    :type population: dict of str and Population
    :param param: Parameters for search
    :type param: dict
    :return: Best individuals
    :rtype: dict

    """

    start_time = time.time()
    param["cache"] = OrderedDict()
    stats: DefaultDict[str, List[Number]] = defaultdict(list)

    ######################
    # Evaluate fitness
    ######################
    population.individuals = evaluate_fitness(
        population.individuals, population.grammar, population.fitness_function, param
    )
    # Set best solution
    population.individuals = sort_population(population.individuals)
    best_ever = population.individuals[0]

    # Print the stats of the populations
    print_stats(0, population.individuals, stats, start_time)

    ######################
    # Generation loop
    ######################
    generation = 1
    while generation < param["generations"]:
        start_time = time.time()

        ##################
        # Selection
        ##################
        parents = tournament_selection(
            population.individuals, param["population_size"], param["tournament_size"]
        )

        ##################
        # Variation. Generate new individual solutions
        ##################
        new_individuals = variation(parents, param, param["population_size"])

        ##################
        # Evaluate fitness
        ##################
        new_individuals = evaluate_fitness(
            new_individuals, population.grammar, population.fitness_function, param
        )

        ##################
        # Replacement. Replace individual solutions in the population
        ##################
        population.individuals = generational_replacement(
            new_individuals,
            population.individuals,
            population_size=param["population_size"],
            elite_size=param["elite_size"],
        )

        # Set best solution. Replacement does not guarantee sorted solutions
        population.individuals = sort_population(population.individuals)
        best_ever = population.individuals[0]

        # Print the stats of the populations
        print_stats(generation, population.individuals, stats, start_time)

        # Increase the generation counter
        generation += 1

    write_run_output(generation, stats, param)

    return best_ever


def place_individuals_on_graph(population: PopulatedGraph) -> PopulatedGraph:
    individuals: List[Individual] = population.individuals
    graph: Dict[str, List[str]] = population.graph

    #  TODO change update rule to work
    if [ind.fitness for ind in individuals] == [DEFAULT_FITNESS] * len(individuals):
        # randomize order of individuals and pair with keys/vertex labels
        random.shuffle(individuals)
        zip_individuals_vertices = zip([*graph], individuals)
        # Create a dictionary from zip object
        population.map_individuals_to_graph = dict(zip_individuals_vertices)
    else:
        # place by fitness
        for key in graph.keys():
            neighbors_and_self = graph[key] + [key]
            individuals_on_neighbors_and_self = population.map_individuals_to_graph[
                neighbors_and_self
            ]
            fitnesses = [ind.fitness for ind in individuals_on_neighbors_and_self]
            sorted_inds = [x for _, x in sorted(zip(fitnesses, individuals_on_neighbors_and_self))]
            population.map_individuals_to_graph[key] = sorted_inds[0]

    return population


def search_loop_spatial(population: PopulatedGraph, param: Dict[str, Any]) -> Individual:
    """Return the best individual from the evolutionary search loop. Assumes
    the population is initially not evaluated.

    :param population: Initial populations for search
    :type population: dict of str and Population
    :param param: Parameters for search
    :type param: dict
    :return: Best individuals
    :rtype: dict

    """

    start_time = time.time()
    param["cache"] = OrderedDict()
    stats: DefaultDict[str, List[Number]] = defaultdict(list)

    ######################
    # Evaluate fitness
    ######################

    # make individuals using grammar
    for ind in population.individuals:
        map_input_with_grammar(ind, population.grammar)
        assert ind.phenotype != ""

    #  place individuals on map
    place_individuals_on_graph(population)

    # choose a "best-ever" [arbitrary]
    # TODO fix best-ever in spatial
    best_ever = population.individuals[0]

    # Print the stats of the populations
    # TODO fix to print spatial stats in a better way
    print_stats(0, population.individuals, stats, start_time)
    write_spatial_output(0, population, param)

    ####
    # generation loop
    ####

    generation = 1
    while generation < param["generations"]:
        start_time = time.time()

        # evaluate fitness of individual @ each node
        population = evaluate_fitness_spatial(population, param)

        # change node-individual mappings
        place_individuals_on_graph(population)

        # Print the stats of the populations
        print_stats(generation, population.individuals, stats, start_time)
        write_spatial_output(generation, population, param)

        # Increase the generation counter
        generation += 1

    write_run_output(generation, stats, param)

    return best_ever


def search_loop_coevolution(
    populations: Dict[str, CoevPopulation], param: Dict[str, Any]
) -> Dict[str, Individual]:
    """Return the best individual from the evolutionary search
    loop.
    :param populations: Initial populations for search
    :type populations: dict of str and Population
    :param param: Parameters for search
    :type param: dict
    :return: Best individuals
    :rtype: dict
    """

    # Evaluate fitness
    param["cache"] = OrderedDict()

    stats_dict: OrderedDict[str, Any] = OrderedDict()  # pylint: disable=unsubscriptable-object
    _best: OrderedDict[str, Individual] = OrderedDict()  # pylint: disable=unsubscriptable-object

    for key, population in populations.items():
        start_time = time.time()
        stats_dict[key] = defaultdict(list)
        stats = stats_dict[key]
        grammar = population.grammar
        fitness_function = population.fitness_function
        adversary = populations[population.adversary]
        for ind in adversary.individuals:
            map_input_with_grammar(ind, adversary.grammar)

        population.individuals = evaluate_fitness_coev(
            population.individuals, grammar, fitness_function, adversary.individuals, param
        )
        # Set best solution
        population.individuals = sort_population(population.individuals)
        _best[key] = population.individuals[0]

        # Print the stats of the populations
        print(key, len(param["cache"]))
        print_stats(0, population.individuals, stats, start_time)

    # Generation loop
    generation = 1
    while generation < param["generations"]:
        if len(param["cache"]) > CACHE_MAX_SIZE:
            param["cache"].clear()

        for key, population in populations.items():
            start_time = time.time()
            stats = stats_dict[key]
            grammar = population.grammar
            fitness_function = population.fitness_function
            adversary = populations[population.adversary]
            for ind in adversary.individuals:
                map_input_with_grammar(ind, adversary.grammar)

            # Selection
            parents = tournament_selection(
                population.individuals, param["population_size"], param["tournament_size"]
            )

            elites = [Individual(_.genome) for _ in population.individuals[: param["elite_size"]]]

            new_individuals = variation(
                parents, param, param["population_size"] - param["elite_size"]
            )

            new_individuals = elites + new_individuals

            # Evaluate fitness
            new_individuals = evaluate_fitness_coev(
                new_individuals, grammar, fitness_function, adversary.individuals, param
            )

            # Replace populations

            # Fitness is relative the adversaries, thus an elite must
            # always be re-evaluated
            population.individuals = generational_replacement(
                new_individuals,
                population.individuals,
                population_size=param["population_size"],
                elite_size=0,
            )

            # Set best solution
            population.individuals = sort_population(population.individuals)
            _best[key] = population.individuals[0]

            # Print the stats of the populations
            print(key, len(param["cache"]))
            print_stats(generation, population.individuals, stats, start_time)

        # Increase the generation counter
        generation += 1

    write_run_output_coev(generation, stats_dict, populations, param)

    best_solution_str = ["%s: %s" % (k, v) for k, v in _best.items()]
    print("Best solution: %s" % (",".join(best_solution_str)))

    return _best


def evaluate(
    individual: Individual,
    opponent_str: str,
    fitness_function: FitnessFunction,
    cache: Dict[str, float],
) -> Individual:
    """Evaluates phenotype in fitness_function function and sets fitness_function.

    :param individual:
    :type individual: Individual
    :param fitness_function: Fitness function
    :type fitness_function: function
    :param cache: Cache for evaluation speed-up
    :type cache: dict
    :return: individual
    :rtype: Individual
    """

    # in the non-coevolution case, the opponent is passed as just a string
    individual.fitness = fitness_function(individual.phenotype, opponent_str, cache)

    assert individual.fitness is not None

    return individual


def evaluate_coev_spatial(
    individual: Individual, fitness_function: Any, inds: List[Individual], cache: Dict[str, float]
) -> Individual:
    """Evaluates phenotype in fitness_function function and sets fitness_function.
    :param individual:
    :type individual: Individual
    :param fitness_function: Fitness function
    :type fitness_function: function
    :param inds: Other individuals
    :type inds: list of Individuals
    :param cache: Cache for evaluation speed-up
    :type cache: dict
    :return: individual
    :rtype: Individual
    """

    individual.fitness = fitness_function.coev(individual.phenotype, inds, cache)

    assert individual.fitness is not None

    return individual


def evaluate_fitness(
    individuals: List[Individual],
    grammar: Grammar,
    fitness_function: FitnessFunction,
    param: Dict[str, Any],
) -> List[Individual]:
    """Perform the fitness evaluation for each individual of the population.

    :param individuals:
    :type individuals: list of Individual
    :param grammar:
    :type grammar: Grammar
    :param fitness_function:
    :type fitness_function: function
    :param param: Other parameters
    :type param: dict
    :return: Evaluated individuals
    :rtype: list of Individuals

    """
    cache = param["cache"]
    n_individuals = len(individuals)
    # Iterate over all the individual solutions
    for ind in individuals:
        map_input_with_grammar(ind, grammar)
        assert ind.phenotype
        if ind.phenotype != "":
            # Execute the fitness function
            evaluate(ind, param["fitness_function"]["opponent"], fitness_function, cache)

    assert n_individuals == len(individuals), "{} != {}".format(n_individuals, len(individuals))

    return individuals


def evaluate_fitness_coev(
    individuals: List[Individual],
    grammar: Grammar,
    fitness_function: FitnessFunction,
    adversaries: List[Individual],
    param: Dict[str, Any],
) -> List[Individual]:
    """Perform the fitness evaluation for each individual of the population.
    :param individuals:
    :type individuals: list of Individual
    :param grammar:
    :type grammar: Grammar
    :param fitness_function:
    :type fitness_function: function
    :param adversaries: Competitors (or collaborators) of individual
    :type adversaries: list of Individuals
    :param param: Other parameters
    :type param: dict
    :return: Evaluated indviduals
    :rtype: list of Individuals
    """

    # TODO efficient caching for parallel evaluation
    cache = param["cache"]

    n_individuals = len(individuals)
    # Iterate over all the individual solutions
    for ind in individuals:
        # TODO map only once
        map_input_with_grammar(ind, grammar)
        assert ind.phenotype
        if ind.phenotype != "":
            # Execute the fitness function
            evaluate_coev_spatial(ind, fitness_function, adversaries, cache)

    assert n_individuals == len(individuals), "%d != %d" % (n_individuals, len(individuals))

    return individuals


def evaluate_fitness_spatial(population: PopulatedGraph, param: Dict[str, Any]) -> PopulatedGraph:
    """Perform the fitness evaluation for each individual of the population.

    :param individuals:
    :type individuals: list of Individual
    :param grammar:
    :type grammar: Grammar
    :param fitness_function:
    :type fitness_function: function
    :param param: Other parameters
    :type param: dict
    :return: Evaluated individuals
    :rtype: list of Individuals

    """
    cache = param["cache"]

    for vertex in population.graph:
        #  calculate fitness of individual at this node
        cur_ind = population.map_individuals_to_graph[vertex]
        neighbor_indices = population.graph[vertex]
        neighbors = population.map_individuals_to_graph[neighbor_indices]
        evaluate_coev_spatial(cur_ind, neighbors, population.fitness_function, cache)

    return population


def variation(parents: List[Individual], param: Dict[str, Any], num_inds: int) -> List[Individual]:
    """
    Vary individual solutions with crossover and mutation operations. Drive the
    search by generating variation of the parent solutions.

    :param parents: Collection of individual solutions
    :type parents: list of Individuals
    :param param: Parameters
    :type param: dict
    :return: Collection of individual solutions
    :rtype: list of Individuals
    """

    assert len(parents) > 1, "{} < 1".format(len(parents))
    assert num_inds > 0, "{} < 0".format(num_inds)

    ###################
    # Crossover
    ###################
    new_individuals: List[Individual] = []
    while len(new_individuals) < num_inds and len(parents) > 1:
        # Select parents
        _parents = random.sample(parents, 2)
        # Generate children by crossing over the parents
        children = onepoint_crossover(_parents[0], _parents[1], param["crossover_probability"])
        # Append the children to the new populations
        for child in children:
            new_individuals.append(child)

    # Select populations size individuals. Handles uneven populations
    # sizes, since crossover returns 2 offspring
    assert len(new_individuals) >= num_inds
    new_individuals = new_individuals[:num_inds]

    ###################
    # Mutation
    ###################
    for i, _ in enumerate(new_individuals):
        new_individuals[i] = int_flip_mutation(new_individuals[i], param["mutation_probability"])

    assert num_inds == len(new_individuals)

    return new_individuals


def int_flip_mutation(individual: Individual, mutation_probability: float) -> Individual:
    """Mutate the individual by randomly choosing a new int with
    probability.

    :param individual:
    :type individual: Individual
    :param mutation_probability: Probability of changing value
    :type mutation_probability: float
    :return: Mutated individual
    :rtype: Individual

    """

    assert Individual.codon_size > 0
    assert 0 <= mutation_probability <= 1.0

    for i in range(len(individual.genome)):
        if random.random() < mutation_probability:
            individual.genome[i] = random.randint(0, Individual.codon_size)
            individual.phenotype = Individual.DEFAULT_PHENOTYPE
            individual.used_input = 0
            individual.fitness = DEFAULT_FITNESS

    return individual


def onepoint_crossover(
    p_0: Individual, p_1: Individual, crossover_probability: float
) -> List[Individual]:
    """Given two individuals, create two children using one-point
    crossover and return them.

    :param p_0: A parent
    :type p_0: Individual
    :param p_1: Another parent
    :type p_1: Individual
    :param crossover_probability: Probability of crossover
    :type crossover_probability: float
    :return: A pair of new individual solutions
    :rtype: list of Individuals

    """
    assert p_0.used_input > 0 and p_1.used_input > 0
    # Get the chromosomes
    c_p_0 = p_0.genome
    c_p_1 = p_1.genome
    # Only within used codons
    max_p_0 = p_0.used_input
    max_p_1 = p_1.used_input

    pt_p_0, pt_p_1 = random.randint(1, max_p_0), random.randint(1, max_p_1)
    # Make new chromosomes by crossover: these slices perform copies
    if random.random() < crossover_probability:
        c_0 = c_p_0[:pt_p_0] + c_p_1[pt_p_1:]
        c_1 = c_p_1[:pt_p_1] + c_p_0[pt_p_0:]
    else:
        c_0 = c_p_0[:]
        c_1 = c_p_1[:]

    individuals = [Individual(c_0), Individual(c_1)]

    return individuals


def sort_population(individuals: List[Individual]) -> List[Individual]:
    """
    Return a list sorted on the fitness value of the individuals in
    the population. Descending order.

    :param individuals: The population of individuals
    :type individuals: list
    :return: Population of individuals sorted by fitness in descending order
    :rtype: list

    """

    # Sort the individual elements on the fitness
    individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)

    return individuals


def tournament_selection(
    population: List[Individual], population_size: int, tournament_size: int
) -> List[Individual]:
    """
    Return individuals from a population by drawing
    `tournament_size` competitors randomly and selecting the best
    of the competitors. `population_size` number of tournaments are
    held.

    :param population: Individuals to draw from
    :type population: list of Individual
    :param population_size: Number of individuals to select
    :type population_size: int
    :param tournament_size: Number of competing individuals
    :type tournament_size: int
    :return: Selected individuals
    :rtype: list of Individuals
    """
    assert tournament_size > 0
    assert tournament_size <= len(population), "{} > {}".format(tournament_size, len(population))

    # Iterate until there are enough tournament winners selected
    winners: List[Individual] = []
    while len(winners) < population_size:
        # Randomly select tournament size individual solutions
        # from the population.
        competitors = random.sample(population, tournament_size)
        # Rank the selected solutions
        competitors = sort_population(competitors)
        # Append the best solution to the winners
        winners.append(competitors[0])

    assert len(winners) == population_size

    return winners


def generational_replacement(
    new_population: List[Individual],
    old_population: List[Individual],
    elite_size: int,
    population_size: int,
) -> List[Individual]:
    """
    Return a new population. The `elite_size` best old_population
    are appended to the new population.

    # TODO the number of calls to sort_population can be reduced

    :param new_population: the new population
    :type new_population: list
    :param old_population: the old population
    :type old_population: list
    :param elite_size: Number of individuals to keep for new population
    :type elite_size: int
    :param population_size: Number of solutions in new population
    :type population_size: int
    :returns: the new population with the best from the old population
    :rtype: list
    """
    assert len(old_population) == len(new_population) == population_size
    assert 0 <= elite_size < population_size

    # Sort the population
    old_population = sort_population(old_population)
    # Append a copy of the elite_size of the old population to
    # the new population.
    for ind in old_population[:elite_size]:
        # TODO is this deep copy redundant
        new_population.append(copy.deepcopy(ind))

    # Sort the new population
    new_population = sort_population(new_population)

    # Set the new population size
    new_population = new_population[:population_size]
    assert len(new_population) == population_size

    return new_population
