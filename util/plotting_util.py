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
    good_files: List[str] = [i for i in files if "fitness_values.json" in i]

    if len(good_files) > 0:
        for _file in good_files:
            file_path: str = os.path.join(in_path, _file)
            plot_name = _file.replace(".json", ".pdf")
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
            plt.savefig(os.path.join(out_path, plot_name))
            plt.show()
            plt.clf()


def plot_population_freqs(out_path: str = ".", in_path: str = ".", title: str = "") -> None:
    """
    Plot the population frequencies over all generations
    """
    files: List[str] = os.listdir(in_path)
    good_files: List[str] = [i for i in files if "solution_values.json" in i]

    if len(good_files) > 0:
        for _file in good_files:
            file_path: str = os.path.join(in_path, _file)
            plot_name = _file.replace(".json", ".pdf")
            with open(file_path, "r") as in_file:
                data = json.load(in_file)

            # collect data
            flat_data: List[str] = [
                strat for sub_list in data["solution_values"] for strat in sub_list
            ]
            legend_items = np.unique(np.array(flat_data))  # automatically sorts
            freq_mat = np.array(
                [
                    [sub_list.count(item) for item in legend_items]
                    for sub_list in data["solution_values"]
                ]
            )
            df = pd.melt(pd.DataFrame(freq_mat, columns=legend_items))
            df["Generations"] = [i for i in range(len(data["solution_values"]))] * len(legend_items)
            df["Proportion of population"] = df["value"] / (
                len(flat_data) / len(data["solution_values"])
            )

            sns.lineplot(
                x="Generations", y="Proportion of population", hue="variable", data=df
            ).set_title(title)
            plt.ylim(-0.08, 1.08)
            plt.savefig(os.path.join(out_path, plot_name))
            plt.show()

            plt.clf()


if __name__ == "__main__":
    plot_fitness()
