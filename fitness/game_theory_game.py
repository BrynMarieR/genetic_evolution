from typing import List, Dict, Tuple, Callable, Any
import json
import inspect


class GameTheoryGame:
    """
    A game theoretic game

    Attributes:
        n_iterations: Number of iterations
        memory_size: Size of history available
    """

    def __init__(
        self,
        n_iterations: int = 1,
        memory_size: int = 1,
        store_stats: bool = False,
        out_file_name: str = "tmp_out.json",
    ) -> None:
        """ Constructor
        """
        self.n_iterations = n_iterations
        self.memory_size = memory_size
        self.store_stats = store_stats
        self.out_file_name = out_file_name

        if self.store_stats:
            with open(self.out_file_name, "w") as out_file:
                json.dump([], out_file)

    def get_payoff(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        raise NotImplementedError("Implement in game")

    # overload in subclass to get different player behavior based on state
    @staticmethod
    def get_move(
        player: Callable[[List[str], int], str],
        state: Dict[str, Dict[str, List[Any]]],
        iteration: int,
        opponent: str,
    ) -> str:
        """ Helper function to get the player move.

        Player is a function that takes the history and current iteration into account
        """
        move = player(state[opponent]["strategy_history"], iteration)
        return move

    def run(
        self, player_1: Callable[[List[str], int], str], player_2: Callable[[List[str], int], str],
    ) -> Tuple[List[Tuple[float, float]], Dict[str, Dict[str, List[Any]]]]:
        """Return the payoff for each iteration of the game.
        """

        # more complicated states may be defined in subclasses
        # e.g., add in concept of ownership as parameter to players that may
        # persist over time for a single player.
        # TODO leverage more complete history
        state: Dict[str, Dict[str, List[Any]]] = {
            "player_1": {"strategy_history": [""] * self.memory_size},
            "player_2": {"strategy_history": [""] * self.memory_size},
        }

        payoffs: List[Tuple[float, float]] = []
        _payoff = self.get_payoff()
        for i in range(self.n_iterations):
            move_1 = GameTheoryGame.get_move(player_1, state, i, opponent="player_2")
            move_2 = GameTheoryGame.get_move(player_2, state, i, opponent="player_2")
            # add player move to history
            if i == 0:
                state["player_1"]["strategy_history"] = [move_1]
                state["player_2"]["strategy_history"] = [move_2]
            else:
                state["player_1"]["strategy_history"].append(move_1)
                state["player_2"]["strategy_history"].append(move_2)
            moves = (move_1, move_2)
            payoffs.append(_payoff[moves])

        if self.store_stats:
            self.dump_stats(player_1, player_2, payoffs, state)

        return payoffs, state

    def dump_stats(
        self,
        player_1: Callable[[List[str], int], str],
        player_2: Callable[[List[str], int], str],
        payoffs: List[Tuple[float, float]],
        state: Dict[str, Dict[str, List[Any]]],
    ) -> None:
        """ Append run statistics to JSON file.

        Note, File IO can be slow.
        """

        data = {
            "player_1": str(inspect.getsourcelines(player_1)[0]),
            "player_2": str(inspect.getsourcelines(player_2)[0]),
            "payoffs": payoffs,
            "history": state,
        }
        with open(self.out_file_name, "r") as in_file:
            json_data = json.load(in_file)

        json_data.append(data)
        with open(self.out_file_name, "w") as out_file:
            json.dump(json_data, out_file)


class PrisonersDilemma(GameTheoryGame):
    """
    Prisoners Dilemma game, see https://en.wikipedia.org/wiki/Prisoner%27s_dilemma

    """

    COOPERATE: str = "C"
    DEFECT: str = "D"
    R: float = 1.0  # Reward
    P: float = 2.0  # Penalty
    S: float = 3.0  # Sucker
    T: float = 0.0  # Temptation
    PAYOFF: Dict[Tuple[str, str], Tuple[float, float]] = {
        (COOPERATE, COOPERATE): (R, R),
        (COOPERATE, DEFECT): (S, T),
        (DEFECT, COOPERATE): (T, S),
        (DEFECT, DEFECT): (P, P),
    }
    DEFAULT_OUT_FILE: str = "ipd_stats.json"

    def get_payoff(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Return payoff for each strategy combination."""
        return PrisonersDilemma.PAYOFF


# TODO  Implement game from 'Ecotypic variation in the asymmetric Hawk-Dove game: when is Bourgeois
#  an evolutionarily stable strategy?', Michael Mesterton-Gibbons
class HawkAndDove(GameTheoryGame):
    """
    Hawk And Dove game, see https://en.wikipedia.org/wiki/Chicken_(game)

    """

    HAWK: str = "H"
    DOVE: str = "D"
    V: float = 2.0
    C: float = 4.0
    PAYOFF: Dict[Tuple[str, str], Tuple[float, float]] = {
        (HAWK, HAWK): ((V - C) / 2.0, (V - C) / 2.0),
        (HAWK, DOVE): (V, 0),
        (DOVE, HAWK): (0, V),
        (DOVE, DOVE): (V / 2.0, V / 2.0),
    }
    DEFAULT_OUT_FILE: str = "hd_stats.json"

    def get_payoff(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Return payoff for each strategy combination."""
        return HawkAndDove.PAYOFF


class IntrusiveHawkAndDoveGame(GameTheoryGame):
    """
    Hawk And Dove game, see 'Ecotypic variation in the asymmetric Hawk-Dove game: when is Bourgeois
    an evolutionarily stable strategy?', Michael Mesterton-Gibbons

    """

    HAWK: str = "H"
    DOVE: str = "D"
    BOUR: str = "B"
    ANTI: str = "X"

    V: float = 2.0
    C: float = 4.0

    PAYOFF: Dict[Tuple[str, str], Tuple[float, float]] = {
        (HAWK, HAWK): ((V - C) / 2.0, (V - C) / 2.0),
        (HAWK, BOUR): ((3 * V - C) / 4.0, (V - C) / 4.0),
        (HAWK, ANTI): ((3 * V - C) / 4.0, (V - C) / 4.0),
        (HAWK, DOVE): (V, 0),
        (BOUR, HAWK): ((V - C) / 4.0, (3 * V - C) / 4.0),
        (BOUR, BOUR): (V / 2.0, V / 2.0),
        (BOUR, ANTI): ((2 * V - C) / 4.0, (2 * V - C) / 4.0),
        (BOUR, DOVE): (3 * V / 4.0, V / 4.0),
        (ANTI, HAWK): ((V - C) / 4.0, (3 * V - C) / 4.0),
        (ANTI, BOUR): ((2 * V - C) / 4.0, (2 * V - C) / 4.0),
        (ANTI, ANTI): (V / 2.0, V / 2.0),
        (ANTI, DOVE): (3 * V / 4.0, V / 4.0),
        (DOVE, HAWK): (0, V),
        (DOVE, BOUR): (V / 4.0, 3 * V / 4.0),
        (DOVE, ANTI): (V / 4.0, 3 * V / 4.0),
        (DOVE, DOVE): (V / 2.0, V / 2.0),
    }
    DEFAULT_OUT_FILE: str = "intrusive_hd_stats.json"

    def get_payoff(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Return payoff for each strategy combination."""
        return IntrusiveHawkAndDoveGame.PAYOFF


class NonIntrusiveHawkAndDoveGame(GameTheoryGame):
    """
    Hawk And Dove game, see 'Ecotypic variation in the asymmetric Hawk-Dove game: when is Bourgeois
    an evolutionarily stable strategy?', Michael Mesterton-Gibbons

    """

    HAWK: str = "H"
    DOVE: str = "D"
    BOUR: str = "B"
    ANTI: str = "X"

    V: float = 2.0
    C: float = 4.0

    PAYOFF: Dict[Tuple[str, str], Tuple[float, float]] = {
        (HAWK, HAWK): ((V - C) / 2.0, (V - C) / 2.0),
        (HAWK, BOUR): ((3 * V - C) / 4.0, (V - C) / 4.0),
        (HAWK, ANTI): ((3 * V - C) / 4.0, (V - C) / 4.0),
        (HAWK, DOVE): (V, 0),
        (BOUR, HAWK): ((V - C) / 4.0, (3 * V - C) / 4.0),
        (BOUR, BOUR): (V / 2.0, V / 2.0),
        (BOUR, ANTI): ((V - C) / 4.0, (3 * V - C) / 4.0),
        (BOUR, DOVE): (V / 2.0, V / 2.0),
        (ANTI, HAWK): ((V - C) / 4.0, (3 * V - C) / 4.0),
        (ANTI, BOUR): ((3 * V - C) / 4.0, (V - C) / 4.0),
        (ANTI, ANTI): (V / 2.0, V / 2.0),
        (ANTI, DOVE): (V, 0),
        (DOVE, HAWK): (0, V),
        (DOVE, BOUR): (V / 2.0, V / 2.0),
        (DOVE, ANTI): (0, V),
        (DOVE, DOVE): (V / 2.0, V / 2.0),
    }
    DEFAULT_OUT_FILE: str = "nonintrusive_hd_stats.json"

    def get_payoff(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Return payoff for each strategy combination."""
        return NonIntrusiveHawkAndDoveGame.PAYOFF
