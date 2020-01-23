import json
from typing import List
import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_fitness(out_path: str = ".", in_path: str = ".",) -> None:
    """
    Plot the fitness per generation
    """
    files: List[str] = os.listdir(in_path)

    for _file in files:
        # TODO: fix path to file, which is not the same for a coev fitness value
        if _file == "donkey_ge_fitness_values.json":
            file_path: str = os.path.join(in_path, _file)
            with open(file_path, "r") as in_file:
                data = json.load(in_file)

            fitness = np.array(data["fitness_values"])
            plt.subplot(1, 1, 1)
            plt.title("Fitness per generation")
            # Best fitness
            ys = fitness[:, 0]
            plt.plot(ys, label="Best fitness")
            # Average fitness per generation
            ys = np.mean(fitness, 1)
            plt.plot(ys, label="Mean fitness")
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.legend()
            plot_name = _file.replace(".json", ".pdf")
            plt.savefig(os.path.join(out_path, plot_name))


def plot_population_freqs(out_path: str = ".", in_path: str = ".", title: str = "") -> None:
    """
    Plot the population frequencies over all generations
    """
    files: List[str] = os.listdir(in_path)

    for _file in files:
        if _file == "donkey_ge_solution_values.json":
            file_path: str = os.path.join(in_path, _file)
            with open(file_path, "r") as in_file:
                data = json.load(in_file)

            # collect data
            flat_data: List[str] = [
                strat for sub_list in data["solution_values"] for strat in sub_list
            ]
            legend_items = set(flat_data)
            freq_mat = np.array(
                [
                    [sub_list.count(item) for item in legend_items]
                    for sub_list in data["solution_values"]
                ]
            )
            df = pd.melt(pd.DataFrame(freq_mat, columns=legend_items))
            df["generation"] = [i for i in range(len(data["solution_values"]))] * len(legend_items)

            sns.lineplot(x="generation", y="value", hue="variable", data=df).set_title(title)
            plot_name = _file.replace(".json", ".pdf")
            plt.savefig(os.path.join(out_path, plot_name))


if __name__ == "__main__":
    plot_fitness()
