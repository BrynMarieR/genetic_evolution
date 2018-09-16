#! /usr/bin/env python
import math
import time

import argparse
import collections
import copy
import os
import random
import re

from fitness import fitness

__author__ = "Erik Hemberg"
"""GE implementation. Bastardization of PonyGP and PonyGE.

TODO make version that does not require numpy...

"""


class Grammar(object):
    """
    Context Free Grammar. Symbols are tuples with (value, type),
    type is Terminal or NonTerminal
    """

    NT = "NT"  # Non Terminal
    T = "T"  # Terminal
    rule_separator = "::="
    production_separator = "|"

    def __init__(self, file_name):
        """Context free grammar.

        :param file_name: grammar file
        :type file_name: str
        """
        self.rules = collections.OrderedDict()
        # TODO use an ordered set
        self.non_terminals = set()
        self.terminals = set()
        self.start_rule = None
        self.file_name = file_name

    def read_bnf_file(self, file_name):
        """Read a grammar file in BNF format. Wrapper for file reading.

        :param file_name: BNF grammar file
        :type file_name: str
        """
        assert file_name.endswith(".bnf")

        # Read the grammar file
        with open(file_name, "r") as in_file:
            lines = in_file.read()

        self.parse_bnf_string(lines)

    def parse_bnf_string(self, lines):
        """Parse a BNF string with REGEXP.

        # TODO use a non-regexp parser
        # TODO does not handle newlines well

        :param lines: BNF grammar
        :type lines: str

        """
        assert lines is not ""
        _lines = lines
        non_terminal_pattern = re.compile(
            r"""(# Group  so `split()` returns all NTs and Ts.
                 # Do not allow space in NTs. Use lookbehind to match "<"
                 # and ">" only if not preceded by backslash.
                 (?<!\\)<\S+?(?<!\\)>
                 )""",
            re.VERBOSE,
        )
        production_separator_regex = re.compile(
            r"""# Use lookbehind to match "|" if not preceded by
                # backslash. `split()` returns only the productions.
                (?<!\\)\|""",
            re.VERBOSE,
        )
        # Left Hand Side(lhs) of rule
        lhs = None

        # Remember last character on line to handle multi line rules
        last_character = None
        lines = lines.split("\n")
        for line in lines:
            line = line.strip()
            if not line.startswith("#") and line != "":
                # Split rules.
                rule_separators = line.count(Grammar.rule_separator)
                assert rule_separators < 2
                if rule_separators == 1:
                    lhs, productions = line.split(Grammar.rule_separator, 1)
                    lhs = lhs.strip()
                    assert len(lhs) > 2
                    assert non_terminal_pattern.search(lhs), "lhs is not a NT: {}".format(lhs)
                    self.non_terminals.add(lhs)
                    if self.start_rule is None:
                        self.start_rule = (lhs, self.NT)

                else:
                    productions = line

                assert productions != "", "{}\n{}\n{}".format(line, lines, _lines)

                # Find terminals and non-terminals
                tmp_productions = []
                production_split = production_separator_regex.split(productions)
                for production in production_split:
                    production = production.strip().replace(r"\|", Grammar.production_separator)
                    tmp_production = []
                    for symbol in non_terminal_pattern.split(production):
                        symbol = symbol.replace(r"\<", "<").replace(r"\>", ">")
                        if len(symbol) == 0:
                            continue
                        elif non_terminal_pattern.match(symbol):
                            tmp_production.append((symbol, self.NT))
                        else:
                            self.terminals.add(symbol)
                            tmp_production.append((symbol, self.T))

                    if tmp_production:
                        tmp_productions.append(tmp_production)

                assert lhs is not None, "No lhs: {}\n{}".format(line, lines)

                # Create a rule
                if lhs not in self.rules:
                    self.rules[lhs] = tmp_productions
                else:
                    if len(production_split) > 1 or last_character == Grammar.production_separator:
                        self.rules[lhs].extend(tmp_productions)
                # TODO does not handle multiline terminals...

                # Remember the last character of the line
                last_character = productions[-1]

    def __str__(self):
        return "T:{}\nNT:{}\nR:{}\nS:{}\n".format(
            self.terminals, self.non_terminals, self.rules, self.start_rule
        )

    def generate_sentence(self, inputs):
        """Map inputs via rules to output sentence (phenotype).

        :param inputs: Inputs used to generate sentence with grammar
        :type inputs: list of int
        :returns: Sentence and number of input used (phenotype)
        :rtype: tuple of str and int
        """
        used_input = 0
        # TODO faster data structure? E.g. queue
        output = []
        # Needed to avoid infinite loops with poorly specified
        # grammars
        cnt = 0
        break_out = len(inputs) * len(self.terminals)
        unexpanded_symbols = [self.start_rule]
        while len(unexpanded_symbols) > 0 and used_input < len(inputs) and cnt < break_out:
            # Expand a production
            current_symbol = unexpanded_symbols.pop(0)
            # Set output if it is a terminal
            if current_symbol[1] != Grammar.NT:
                output.append(current_symbol[0])
            else:
                production_choices = self.rules[current_symbol[0]]
                # Select a production
                current_production = inputs[used_input] % len(production_choices)
                # Use an input if there was more then 1 choice
                if len(production_choices) > 1:
                    used_input += 1

                # Derivation order is left to right(depth-first)
                unexpanded_symbols = production_choices[current_production] + unexpanded_symbols

            cnt += 1

        # Not fully expanded
        if len(unexpanded_symbols) > 0:
            output = None
        else:
            output = "".join(output)

        return output, used_input


class Population(object):
    """A population container"""

    def __init__(self, fitness_function, grammar, individuals):
        """Container for a population.

        TODO use a data structure that maintains order of individuals based on
        fitness to reduce sorting

        :param fitness_function:
        :type fitness_function: function
        :param grammar:
        :type grammar: Grammar
        :param individuals:
        :type individuals: list of Individual
        """
        self.fitness_function = fitness_function
        self.grammar = grammar
        self.individuals = individuals

    def __str__(self):
        individuals = "\n".join(map(str, self.individuals))
        _str = "{} {} \n{}".format(str(self.fitness_function), self.grammar.file_name, individuals)

        return _str


class Individual(object):
    """A GE individual

    :param codon_size: Max integer value for an input element
    :type codon_size: int
    :param max_length: Length of input
    :type max_length: int

    """

    codon_size = None
    max_length = None

    def __init__(self, genome):
        """

        :param genome: Input representation
        :type genome: list of int or None
        """
        assert Individual.max_length > 0, "max_length {}".format(Individual.max_length)
        assert Individual.codon_size > 0, "codon_size {}".format(Individual.codon_size)

        if genome is None:
            self.genome = [
                random.randint(0, Individual.codon_size) for _ in range(Individual.max_length)
            ]
        else:
            self.genome = genome

        self.fitness = None
        self.phenotype = None
        self.used_input = None

    def get_fitness(self):
        return self.fitness

    def __str__(self):
        return "Ind: {0}; {1}".format(str(self.phenotype), self.get_fitness())


def map_input_with_grammar(individual, grammar):
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
    phenotype = None
    n_inputs_used = None
    while phenotype is None and cnt < break_out:
        phenotype, n_inputs_used = grammar.generate_sentence(individual.genome)
        if phenotype is None:
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
    if phenotype is None:
        raise ValueError("Phenotype is None")

    individual.used_input = n_inputs_used

    return individual


def evaluate(individual, fitness_function=None, cache=None):
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

    individual.fitness = fitness_function(individual.phenotype, cache)

    assert individual.fitness is not None

    return individual


def initialise_population(size):
    """Create a population of Individuals of the given size.

    :param size: Number of individuals to generate
    :type size: int
    :return: Randomly generated individuals
    :rtype: list of Individual
    """
    assert size > 0

    individuals = [Individual(None) for _ in range(size)]

    return individuals


def evaluate_wrapper(solution):
    individual = solution["individual"]
    fitness_function = solution["fitness_function"]
    cache = solution["cache"]
    individual = evaluate(individual, fitness_function, cache)

    return {"individual": individual}


def evaluate_fitness(individuals, grammar, fitness_function, param=None):
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
        # Execute the fitness function
        evaluate(ind, fitness_function, cache)

    assert n_individuals == len(individuals), "{} != {}".format(n_individuals, len(individuals))

    return individuals


def variation(parents, param):
    """
    Vary individual solutions with crossover and mutation oeprations. Drive the
    search by generating variation of the parent solutions.

    :param parents: Collection of individual solutions
    :type parents: list of Individuals
    :param param: Parameters
    :type param: dict
    :return: Collection of individual solutions
    :rtype: list of Individuals
    """

    assert len(parents) > 1, "{} < 1".format(len(parents))

    ###################
    # Crossover
    ###################
    new_individuals = []
    while len(new_individuals) < param["population_size"] and len(parents) > 1:
        # Select parents
        _parents = random.sample(parents, 2)
        # Generate children by crossing over the parents
        children = onepoint_crossover(_parents[0], _parents[1], param["crossover_probability"])
        # Append the children to the new populations
        for child in children:
            new_individuals.append(child)

    # Select populations size individuals. Handles uneven populations
    # sizes, since crossover returns 2 offspring
    assert len(new_individuals) >= param["population_size"]
    new_individuals = new_individuals[: param["population_size"]]

    ###################
    # Mutation
    ###################
    for i in range(len(new_individuals)):
        new_individuals[i] = int_flip_mutation(new_individuals[i], param["mutation_probability"])

    assert param["population_size"] == len(new_individuals)

    return new_individuals


def search_loop(population, param):
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
    param["cache"] = collections.OrderedDict()
    stats = collections.defaultdict(list)

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
        new_individuals = variation(parents, param)

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


def write_run_output(generation, stats, param):
    """Write run stats to files.

    :param generation: Generation number
    :type generation: int
    :param stats: Collected statistics of run
    :type stats: dict
    :param param: Parameters
    :type param: dict
    """
    _hist = collections.defaultdict(int)
    for k, v in param["cache"].items():
        # TODO better key?
        _hist[str(v)] += 1

    print(
        "Cache entries:{} Total Fitness Evaluations:{} Fitness Values:{}".format(
            len(param["cache"].keys()),
            generation * param["population_size"] ** 2,
            len(_hist.keys()),
        )
    )

    out_file_name = "donkey_ge"
    if "output_dir" in param:
        output_dir = param["output_dir"]
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        out_file_name = os.path.join(output_dir, out_file_name)

    _out_file_name = "{}_settings.out".format(out_file_name)
    with open(_out_file_name, "w") as out_file:
        for k, v in param.items():
            if k != "cache":
                out_file.write("{}: {}\n".format(k, str(v)))

    for k, v in stats.items():
        _out_file_name = "{}_{}.csv".format(out_file_name, k)
        with open(_out_file_name, "w") as out_file:
            for line in v:
                if k == "solution_values":
                    out_file.write("{}\n".format(";".join(map(str, line))))
                else:
                    out_file.write("{}\n".format(",".join(map(str, line))))


def print_stats(generation, individuals, stats, start_time):
    """
    Print the statistics for the generation and population.

    :param generation: generation number
    :type generation: int
    :param individuals: population to get statistics for
    :type individuals: list
    :param stats: Collected statistics of run
    :type stats: dict
    :param start_time: Start time
    :type start_time: time.time
    """

    def get_ave_and_std(values):
        """
        Return average and standard deviation.

        :param values: Values to calculate on
        :type values: list
        :returns: Average and Standard deviation of the input values
        :rtype: tuple
        """
        _ave = float(sum(values)) / len(values)
        _std = math.sqrt(float(sum([(value - _ave) ** 2 for value in values])) / len(values))
        return _ave, _std

    # Make sure individuals are sorted
    individuals = sort_population(individuals)
    # Get the fitness values
    fitness_values = [i.get_fitness() for i in individuals]
    # Get the number of nodes
    size_values = [i.used_input for i in individuals]
    # Get the max length
    length_values = [len(i.genome) for i in individuals]
    # Get average and standard deviation of fitness
    ave_fit, std_fit = get_ave_and_std(fitness_values)
    # Get average and standard deviation of size
    ave_size, std_size = get_ave_and_std(size_values)
    # Get average and standard deviation of max length
    ave_length, std_length = get_ave_and_std(length_values)
    # Print the statistics
    print(
        "Gen:{} t:{:.3f} fit_ave:{:.2f}+-{:.3f} size_ave:{:.2f}+-{:.3f} "
        "length_ave:{:.2f}+-{:.3f} {}".format(
            generation,
            time.time() - start_time,
            ave_fit,
            std_fit,
            ave_size,
            std_size,
            ave_length,
            std_length,
            individuals[0],
        )
    )

    stats["fitness_values"].append(fitness_values)
    stats["size_values"].append(size_values)
    stats["length_values"].append(length_values)
    stats["solution_values"].append([_.phenotype for _ in individuals])


def int_flip_mutation(individual, mutation_probability):
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
            individual.phenotype = None
            individual.used_input = None
            individual.fitness = None

    return individual


def tournament_selection(population, population_size, tournament_size):
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
    winners = []
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


def onepoint_crossover(p_0, p_1, crossover_probability):
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


def sort_population(individuals):
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


def generational_replacement(new_population, old_population, elite_size, population_size):
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


def parse_arguments():
    """
    Returns a dictionary of the default parameters, or the ones set by
    commandline arguments.

    :return: parameters for the
    :rtype: dict
    """
    # Command line arguments
    parser = argparse.ArgumentParser(description="Run donkey_ge")
    # Population size
    parser.add_argument(
        "-p",
        "--population_size",
        type=int,
        default=4,
        dest="population_size",
        help="population size",
    )
    # Size of an individual
    parser.add_argument(
        "-m", "--max_length", type=int, default=3, dest="max_length", help="Max length"
    )
    # Size of an element in input(genotype)
    parser.add_argument(
        "-c",
        "--integer_input_element_max",
        type=int,
        default=127,
        dest="codon_size",
        help="Input element max value",
    )
    # Number of elites.
    parser.add_argument(
        "-e",
        "--elite_size",
        type=int,
        default=1,
        dest="elite_size",
        help="elite size. The number of top ranked solution from "
        "the old population transferred to the new "
        "population",
    )
    # Generations
    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        default=2,
        dest="generations",
        help="number of generations. Number of times the search " "loop is iterated",
    )
    # Tournament size
    parser.add_argument(
        "--ts",
        "--tournament_size",
        type=int,
        default=2,
        dest="tournament_size",
        help="tournament size",
    )
    # Random seed.
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        dest="seed",
        help="Seed number for the random number generator. Use "
        "the same seed and settings to replicate results.",
    )
    # Probability of crossover
    parser.add_argument(
        "--cp",
        "--crossover_probability",
        type=float,
        dest="crossover_probability",
        default=0.8,
        help="crossover probability",
    )
    # Probability of mutation
    parser.add_argument(
        "--mp",
        "--mutation_probability",
        type=float,
        dest="mutation_probability",
        default=0.1,
        help="mutation probability",
    )
    # Grammar files
    parser.add_argument(
        "-b",
        "--bnf_grammar",
        type=str,
        dest="bnf_grammar",
        default="grammars/symbolic_regression.bnf",
        help="bnf grammar",
    )
    # Fitness function
    parser.add_argument(
        "-f",
        "--fitness_function",
        type=str,
        dest="fitness_function",
        default=None,
        help="fitness function",
    )

    # Parse the command line arguments
    options, _args = parser.parse_args()
    return vars(options)


def run(param):
    """
    Return the best solution. Create an initial
    population. Perform an evolutionary search.

    :param param: parameters for pony gp
    :type param: dict
    :returns: Best solution
    :rtype: dict
    """

    start_time = time.time()

    # Set random seed if not 0 is passed in as the seed
    if "seed" not in param.keys():
        param["seed"] = int(time.time())

    random.seed(param["seed"])
    print("Setting random seed: {} {:.5f}".format(param["seed"], random.random()))

    # Print settings
    print("donkey_ge settings:", param)

    assert param["population_size"] > 1
    assert param["generations"] > 0
    assert param["max_length"] > 0
    assert param["integer_input_element_max"] > 0
    assert param["seed"] > -1
    assert param["tournament_size"] <= param["population_size"]
    assert param["elite_size"] < param["population_size"]
    assert 0.0 <= param["crossover_probability"] <= 1.0
    assert 0.0 <= param["mutation_probability"] <= 1.0

    ###########################
    # Create initial population
    ###########################
    grammar = Grammar(param["bnf_grammar"])
    grammar.read_bnf_file(grammar.file_name)
    fitness_function = fitness.get_fitness_function(param["fitness_function"])
    # These are parameters since defaults are dangerous
    # TODO make clearer
    Individual.max_length = param["max_length"]
    Individual.codon_size = param["integer_input_element_max"]
    individuals = initialise_population(param["population_size"])

    population = Population(fitness_function, grammar, individuals)

    ###########################
    # Evolutionary search
    ###########################
    # TODO revise for stability
    best_ever = search_loop(population, param)

    # Display results
    print("Time: {:.3f} Best solution:{}".format(time.time() - start_time, best_ever))

    return best_ever


if __name__ == "__main__":
    args = parse_arguments()
    run(args)