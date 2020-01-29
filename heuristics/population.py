#! /usr/bin/env python
import random
from typing import List, Any, Optional

from heuristics.grammar import Grammar
from fitness.fitness import DEFAULT_FITNESS


class Individual(object):
    """A GE individual

    Attributes:
        codon_size: Max integer value for an input element
        max_length: Length of input
        DEFAULT_PHENOTYPE:

    """

    codon_size: int = -1
    max_length: int = -1
    DEFAULT_PHENOTYPE = ""

    def __init__(self, genome: Optional[List[int]]) -> None:
        """

        :param genome: Input representation
        :type genome: list of int or None
        """
        assert Individual.max_length > 0, "max_length {}".format(Individual.max_length)
        assert Individual.codon_size > 0, "codon_size {}".format(Individual.codon_size)

        if genome is None:
            self.genome: List[int] = [
                random.randint(0, Individual.codon_size) for _ in range(Individual.max_length)
            ]
        else:
            self.genome = genome

        self.fitness: float = DEFAULT_FITNESS
        self.phenotype: str = Individual.DEFAULT_PHENOTYPE
        self.used_input: int = 0

    def get_fitness(self) -> float:
        """
        Return individual fitness
        """
        return self.fitness

    def __str__(self) -> str:
        return "Ind: {0}; {1}".format(str(self.phenotype), self.get_fitness())


class Population(object):
    """A population container

    Attributes:
        fitness_function:
        grammar:
        individuals:
    """

    def __init__(
        self, fitness_function: Any, grammar: Grammar, individuals: List[Individual]
    ) -> None:
        """Container for a population.

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

    def __str__(self) -> str:
        individuals = "\n".join(map(str, self.individuals))
        _str = "{} {} \n{}".format(str(self.fitness_function), self.grammar.file_name, individuals)

        return _str


class CoevPopulation(Population):
    """A population container"""

    def __init__(
        self,
        fitness_function: Any,
        grammar: Grammar,
        adversary: str,
        name: str,
        individuals: List[Individual],
    ) -> None:
        """Container for a population.
        :param fitness_function:
        :type fitness_function: function
        :param grammar:
        :type grammar: Grammar
        :param adversary:
        :type adversary: str
        :param name:
        :type name: str
        :param individuals:
        :type individuals: list of Individual
        """
        super(CoevPopulation, self).__init__(fitness_function, grammar, individuals)
        self.adversary = adversary
        self.name = name

    def clone(self) -> Population:
        clone = CoevPopulation(
            self.fitness_function, self.grammar, self.adversary, self.name, self.individuals
        )
        return clone

    def __str__(self) -> str:
        individuals = "\n".join(map(str, self.individuals))
        _str = "{} {} {} {}\n{}".format(
            str(self.fitness_function),
            self.grammar.file_name,
            self.adversary,
            self.name,
            individuals,
        )

        return _str


class PopulatedGraph(Population):
    def __init__(
        self,
        graph: Any,
        map_individuals_to_graph: Any,
        fitness_function: Any,
        grammar: Grammar,
        individuals: List[Individual],
    ):
        Population.__init__(self, fitness_function, grammar, individuals)
        Population.__init__(self, fitness_function, grammar, individuals)
        self.graph = graph
        self.map_individuals_to_graph = map_individuals_to_graph
