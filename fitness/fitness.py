"""Module for the fitness functions. A fitness function serves as an interface between
the engagement environment and the search heuristic. It provides the engagement
environment with the actions. It computes a fitness score based on the measurements
from the engagement environment.
"""
from typing import List, Dict, Any, Tuple, Callable

from fitness.game_theory_game import (
    PrisonersDilemma,
    HawkAndDove,
    IntrusiveHawkAndDoveGame,
    NonIntrusiveHawkAndDoveGame,
)


# global vars and helpers
DEFAULT_FITNESS: float = -float("inf")


def mean(values: List[float]) -> float:
    """
    Return the mean of the values.
    """
    return sum(values) / len(values)


class FitnessFunction(object):
    """
    Fitness function abstract class
    """

    def __call__(self, fcn_str: str, strat_str: str, cache: Dict[str, float]) -> float:
        raise NotImplementedError("Define in subclass")


class IteratedPrisonersDilemma(FitnessFunction):
    """
    Iterated Prisoners Dilemma fitness function

    Note: There are faster ways of evaluating, this is for demonstration purposes.

    Attributes:
        n_iterations: Number of iterations of the Prisoners Dilemma
        prisoners_dilemma: Prisoners Dilemma
        opponent: Strategy of opponent
        player: Strategy of player
    """

    def __init__(self, param: Dict[str, Any]) -> None:
        """ Initialize object
        """
        self.n_iterations = param["n_iterations"]
        self.prisoners_dilemma = PrisonersDilemma(self.n_iterations)
        self.opponent: Callable[[List[str], int], str] = eval(  # pylint: disable=eval-used
            param["opponent"]
        )
        self.player: Callable[[List[str], int], str] = lambda x, y: ""

    def __call__(self, fcn_str: str, strat_str: str, cache: Dict[str, float]) -> float:
        """ Evaluate the strategy against the opponent and return fitness.
        """
        key: str = "{}-{}".format(fcn_str, strat_str)
        if key in cache:
            fitness: float = cache[key]
        else:
            self.player = eval(fcn_str)  # pylint: disable=eval-used
            self.opponent = eval(strat_str)  # pylint: disable=eval-used
            payoff, _ = self.prisoners_dilemma.run(self.player, self.opponent)
            fitness = IteratedPrisonersDilemma.get_fitness(payoff)
            cache[key] = fitness

        return fitness

    @staticmethod
    def get_fitness(payoffs: List[Tuple[float, float]]) -> float:
        """ Fitness is the mean of the payoff
        """
        fitness: float = mean([_[0] for _ in payoffs])
        return fitness

    def coev(self, fcn_str: str, strategies: List[Any], cache: Dict[str, float]) -> float:
        """ Evaluate one strategy against multiple strategies and mean expected utility (fitness).
        """
        fitnesses: List[float] = [DEFAULT_FITNESS] * len(strategies)
        for i, strategy in enumerate(strategies):
            fitnesses[i] = self.__call__(fcn_str, strategy.phenotype, cache)

        # Mean Expected Utility
        fitness: float = mean(fitnesses)
        return fitness


class IteratedHawkAndDove(FitnessFunction):
    """
    Hawk and Dove fitness function

    Note: There are faster ways of evaluating, this is for demonstration purposes.

    Attributes:
        n_iterations: Number of iterations of the game
        hawk_and_dove: Hawk And Dove game
        opponent: Strategy of opponent
        player: Strategy of player
    """

    def __init__(self, param: Dict[str, Any]) -> None:
        """ Initialize object
        """
        self.n_iterations = param["n_iterations"]
        self.hawk_and_dove = HawkAndDove(self.n_iterations)
        self.opponent = eval(param["opponent"])  # pylint: disable=eval-used
        self.player = lambda x, y: ""

    def __call__(self, fcn_str: str, strat_str: str, cache: Dict[str, float]) -> float:
        """ Evaluate the strategy against the opponent and return fitness.
        """
        key: str = "{}-{}".format(fcn_str, strat_str)
        if key in cache:
            fitness: float = cache[key]
        else:
            self.player = eval(fcn_str)  # pylint: disable=eval-used
            self.opponent = eval(strat_str)  # pylint: disable=eval-used
            payoff, _ = self.hawk_and_dove.run(self.player, self.opponent)
            fitness = IteratedHawkAndDove.get_fitness(payoff)
            cache[key] = fitness

        return fitness

    @staticmethod
    def get_fitness(payoffs: List[Tuple[float, float]]) -> float:
        """ Fitness is the mean of the payoff
        """
        fitness: float = mean([_[0] for _ in payoffs])
        return fitness

    def coev(self, fcn_str: str, strategies: List[Any], cache: Dict[str, float]) -> float:
        """ Evaluate one strategy against multiple strategies and mean expected utility (fitness).
        """
        fitnesses: List[float] = [DEFAULT_FITNESS] * len(strategies)
        for i, strategy in enumerate(strategies):
            fitnesses[i] = self.__call__(fcn_str, strategy.phenotype, cache)

        # Mean Expected Utility
        fitness: float = mean(fitnesses)
        return fitness


class IntrusiveHawkAndDove(FitnessFunction):
    """
    Intrusive Hawk and Dove fitness function

    Note: There are faster ways of evaluating, this is for demonstration purposes.

    Attributes:
        n_iterations: Number of iterations of the game
        hawk_and_dove: Hawk And Dove game
        opponent: Strategy of opponent
        player: Strategy of player
    """

    def __init__(self, param: Dict[str, Any]) -> None:
        """ Initialize object
        """
        self.n_iterations = param["n_iterations"]
        self.hawk_and_dove = IntrusiveHawkAndDoveGame(self.n_iterations)
        self.opponent = eval(param["opponent"])  # pylint: disable=eval-used
        self.player = lambda x, y: ""

    def __call__(self, fcn_str: str, strat_str: str, cache: Dict[str, float]) -> float:
        """ Evaluate the strategy against the opponent and return fitness.
        """
        key: str = "{}-{}".format(fcn_str, strat_str)
        if key in cache:
            fitness: float = cache[key]
        else:
            self.player = eval(fcn_str)  # pylint: disable=eval-used
            self.opponent = eval(strat_str)  # pylint: disable=eval-used
            payoff, _ = self.hawk_and_dove.run(self.player, self.opponent)
            fitness = IntrusiveHawkAndDove.get_fitness(payoff)
            cache[key] = fitness

        return fitness

    @staticmethod
    def get_fitness(payoffs: List[Tuple[float, float]]) -> float:
        """ Fitness is the mean of the payoff
        """
        fitness: float = mean([_[0] for _ in payoffs])
        return fitness

    def coev(self, fcn_str: str, strategies: List[Any], cache: Dict[str, float]) -> float:
        """ Evaluate one strategy against multiple strategies and mean expected utility (fitness).
        """
        fitnesses: List[float] = [DEFAULT_FITNESS] * len(strategies)
        for i, strategy in enumerate(strategies):
            fitnesses[i] = self.__call__(fcn_str, strategy.phenotype, cache)

        # Mean Expected Utility
        fitness: float = mean(fitnesses)
        return fitness


class NonIntrusiveHawkAndDove(FitnessFunction):
    """
    NonIntrusive Hawk and Dove fitness function

    Note: There are faster ways of evaluating, this is for demonstration purposes.

    Attributes:
        n_iterations: Number of iterations of the game
        hawk_and_dove: Hawk And Dove game
        opponent: Strategy of opponent
        player: Strategy of player
    """

    def __init__(self, param: Dict[str, Any]) -> None:
        """ Initialize object
        """
        self.n_iterations = param["n_iterations"]
        self.hawk_and_dove = NonIntrusiveHawkAndDoveGame(self.n_iterations)
        self.opponent = eval(param["opponent"])  # pylint: disable=eval-used
        self.player = lambda x, y: ""

    def __call__(self, fcn_str: str, strat_str: str, cache: Dict[str, float]) -> float:
        """ Evaluate the strategy against the opponent and return fitness.
        """

        key: str = "{}-{}".format(fcn_str, strat_str)
        if key in cache:
            fitness: float = cache[key]
        else:
            self.player = eval(fcn_str)  # pylint: disable=eval-used
            self.opponent = eval(strat_str)  # pylint: disable=eval-used
            payoff, _ = self.hawk_and_dove.run(self.player, self.opponent)
            fitness = NonIntrusiveHawkAndDove.get_fitness(payoff)
            cache[key] = fitness

        return fitness

    @staticmethod
    def get_fitness(payoffs: List[Tuple[float, float]]) -> float:
        """ Fitness is the average of the payoff
        """
        fitness: float = mean([_[0] for _ in payoffs])
        return fitness

    def coev(self, fcn_str: str, strategies: List[Any], cache: Dict[str, float]) -> float:
        """ Evaluate one strategy against multiple strategies and mean expected utility (fitness).
        """
        fitnesses: List[float] = [DEFAULT_FITNESS] * len(strategies)
        for i, strategy in enumerate(strategies):
            fitnesses[i] = self.__call__(fcn_str, strategy.phenotype, cache)

        # Mean Expected Utility
        fitness: float = mean(fitnesses)
        return fitness


if __name__ == "__main__":
    pass
