import math
import time
import json
import os
import collections
from numbers import Number
from typing import Any, List, Dict, DefaultDict, Sequence, Tuple

import heuristics.population as hpop


def print_cache_stats(generation: int, param: Dict[str, Any]) -> None:
    _hist: DefaultDict[str, int] = collections.defaultdict(int)
    for v in param["cache"].values():
        _hist[str(v)] += 1

    print(
        "Cache:{} Cache entries:{} Total Fitness Evaluations:{} Fitness Values:{}".format(
            param["cache"],
            len(param["cache"].keys()),
            generation * param["population_size"] ** 2,
            len(_hist.keys()),
        )
    )


def get_out_file_name(out_file_name: str, param: Dict[str, Any]) -> str:
    if "output_dir" in param:
        output_dir = param["output_dir"]
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        out_file_name = os.path.join(output_dir, out_file_name)
    return out_file_name


def write_run_output(
    generation: int, stats: Dict[str, List[Number]], param: Dict[str, Any]
) -> None:
    """Write run stats to files.

    :param generation: Generation number
    :type generation: int
    :param stats: Collected statistics of run
    :type stats: dict
    :param param: Parameters
    :type param: dict
    """
    print_cache_stats(generation, param)
    out_file_name = get_out_file_name("donkey_ge", param)
    _out_file_name = "{}_settings.json".format(out_file_name)
    with open(_out_file_name, "w") as out_file:
        _settings: Dict[str, Any] = {}
        for k, v in param.items():
            if k != "cache":
                _settings[k] = v

        json.dump(_settings, out_file, indent=1)

    for k, v in stats.items():
        _out_file_name = "{}_{}.json".format(out_file_name, k)
        with open(_out_file_name, "w") as out_file:
            json.dump({k: v}, out_file, indent=1)


def write_run_output_coev(
    generation: int,
    stats_dict: Dict[str, Dict[str, List[Number]]],
    populations: Dict[str, hpop.CoevPopulation],
    param: Dict[str, Any],
) -> None:
    """Write run stats to files.
    :param generation: Generation number
    :type generation: int
    :param stats_dict: Collected statistics of run
    :type stats_dict: dict
    :param populations: Populations
    :type populations: dict of str and Population
    :param param: Parameters
    :type param: dict
    """
    print_cache_stats(generation, param)
    out_file_name = get_out_file_name("donkey_ge_coev", param)
    _out_file_name = "%s_settings.json" % out_file_name
    with open(_out_file_name, "w") as out_file:
        for k, v in param.items():
            if k != "cache":
                json.dump({k: v}, out_file, indent=1)

    for key in populations.keys():
        stats = stats_dict[key]
        for k, v in stats.items():
            _out_file_name = "%s_%s_%s.json" % (out_file_name, key, k)
            with open(_out_file_name, "w") as out_file:
                json.dump({k: v}, out_file, indent=1)


def print_stats(
    generation: int,
    individuals: List[hpop.Individual],
    stats: Dict[str, List[Any]],
    start_time: float,
) -> None:
    """
    Print the statistics for the generation and population.

    :param generation: generation number
    :type generation: int
    :param individuals: population to get statistics for
    :type individuals: list
    :param stats: Collected statistics of run
    :type stats: dict
    :param start_time: Start time
    :type start_time: float
    """

    def get_ave_and_std(values: Sequence[float]) -> Tuple[float, float]:
        """
        Return average and standard deviation.

        :param values: Values to calculate on
        :type values: list
        :returns: Average and Standard deviation of the input values
        :rtype: tuple
        """
        _ave: float = float(sum(values)) / float(len(values))
        _std: float = math.sqrt(float(sum([(value - _ave) ** 2 for value in values])) / len(values))
        return _ave, _std

    # Make sure individuals are sorted
    individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
    # Get the fitness values
    fitness_values: Sequence[float] = [i.get_fitness() for i in individuals]
    # Get the number of nodes
    size_values: Sequence[float] = [float(i.used_input) for i in individuals]
    # Get the max length
    length_values: Sequence[float] = [float(len(i.genome)) for i in individuals]
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
