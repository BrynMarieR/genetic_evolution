#! /usr/bin/env python
import argparse
import time
import random
from typing import Union, Dict, Any, Tuple
from collections import OrderedDict
from numbers import Number

from fitness.fitness import FitnessFunction
from heuristics.population import Individual, Population, CoevPopulation
from heuristics.grammar import Grammar
from heuristics.ge_helpers import initialise_population, search_loop, search_loop_coevolution

__author__ = "Bryn Reinstadler, Erik Hemberg"
"""GE implementation. Bastardization of PonyGP and PonyGE.
"""


def get_fitness_function(param: Dict[str, str]) -> FitnessFunction:
    """Returns fitness function object.

    Used to construct fitness functions from the configuration parameters

    :param param: Fitness function parameters
    :type param: dict
    :return: Fitness function
    :rtype: Object
    """
    from fitness.fitness import (
        IteratedPrisonersDilemma,
        IteratedHawkAndDove,
        IntrusiveHawkAndDove,
        NonIntrusiveHawkAndDove,
    )

    name = param["name"]
    fitness_function: FitnessFunction
    if name == "IteratedPrisonersDilemma":
        fitness_function = IteratedPrisonersDilemma(param)
    elif name == "IteratedHawkAndDove":
        fitness_function = IteratedHawkAndDove(param)
    elif name == "IntrusiveHawkAndDove":
        fitness_function = IntrusiveHawkAndDove(param)
    elif name == "NonIntrusiveHawkAndDove":
        fitness_function = NonIntrusiveHawkAndDove(param)
    else:
        raise BaseException("Unknown fitness function: {}".format(name))

    return fitness_function


def run(param: Dict[str, Any], coev: bool) -> Dict[str, Individual]:
    """
    Return the best solution. Create an initial
    population. Perform an evolutionary search.

    :param param: parameters for pony gp
    :type param: dict
    :returns: Best solution
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
    # Create Population + Evolutionary search
    ###########################

    if coev:
        populations: OrderedDict[str, Any] = OrderedDict()  # pylint: disable=unsubscriptable-object
        for key in param["populations"].keys():
            p_dict = param["populations"][key]
            grammar = Grammar(p_dict["bnf_grammar"])
            grammar.read_bnf_file(grammar.file_name)
            fitness_function = get_fitness_function(p_dict["fitness_function"])
            adversary = p_dict["adversary"]
            Individual.max_length = param["max_length"]
            Individual.codon_size = param["integer_input_element_max"]
            individuals = initialise_population(param["population_size"])
            population_coev = CoevPopulation(fitness_function, grammar, adversary, key, individuals)
            populations[key] = population_coev
            best_overall_solution_coev = search_loop_coevolution(populations, param)

        # Display results
        print(
            "Time: {:.3f} Best solution:{}".format(
                time.time() - start_time, best_overall_solution_coev
            )
        )

        return best_overall_solution_coev
    else:
        grammar = Grammar(param["bnf_grammar"])
        grammar.read_bnf_file(grammar.file_name)
        fitness_function = get_fitness_function(param["fitness_function"])
        # These are parameters since defaults are dangerous
        # TODO make clearer
        Individual.max_length = param["max_length"]
        Individual.codon_size = param["integer_input_element_max"]
        individuals = initialise_population(param["population_size"])

        population = Population(fitness_function, grammar, individuals)
        best_overall_solution = search_loop(population, param)

        # Display results
        print(
            "Time: {:.3f} Best solution:{}".format(time.time() - start_time, best_overall_solution)
        )

        return {"best": best_overall_solution}


def parse_arguments() -> Tuple[Dict[str, Union[str, bool, Number]], bool]:
    """
    Returns a dictionary of the default parameters, or the ones set by
    commandline arguments.

    :return: parameters for the
    :rtype: dict
    """
    # Command line arguments
    parser = argparse.ArgumentParser(description="Run ge_run")
    # Population size
    parser.add_argument(
        "--coev",
        "--coev",
        type=bool,
        default=False,
        dest="coevolution",
        help="Flag to do coevolution rather than simple evolution",
    )
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
    return vars(options), vars(options)["coev"]


if __name__ == "__main__":
    ARGS, coev_arg = parse_arguments()
    run(ARGS, coev_arg)