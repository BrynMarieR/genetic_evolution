# script to create default graphs
# graphs are just dictionaries with string keys and
# lists of strings of neighbors
# currently, all graphs are undirected
from typing import List
import json


def create_default_graphs() -> None:
    # generate a lattice graph with 100 vertices
    row_keys: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    col_keys = row_keys
    all_keys = [row + col for row in row_keys for col in col_keys]
    neighbor_lists_lattice = [
        [
            str(x).zfill(2)
            for x in [key_index + 1, key_index - 1, key_index + 10, key_index - 10]
            if (0 <= x < 100)
        ]
        for key_index in range(len(all_keys))
    ]
    lattice_graph_dict = dict(zip(all_keys, neighbor_lists_lattice))
    with open("lattice_graph.json", "w") as out_file:
        json.dump(lattice_graph_dict, out_file, indent=1)

    # generate a path graph with 100 vertices
    neighbor_lists_path = [
        [str(x).zfill(2) for x in [key_index + 1, key_index - 1] if 0 <= x < 100]
        for key_index in range(len(all_keys))
    ]
    path_graph_dict = dict(zip(all_keys, neighbor_lists_path))
    with open("path_graph.json", "w") as out_file:
        json.dump(path_graph_dict, out_file, indent=1)

    # generate a small (17-vertex) forest graph
    forest_keys = [str(x).zfill(2) for x in list(range(17))]
    neighbor_lists_forest = [
        ["01", "05", "09", "13"],
        ["02", "03", "04"],
        ["01"],
        ["01"],
        ["01"],
        ["06", "07", "08"],
        ["05"],
        ["05"],
        ["05"],
        ["10", "11", "12"],
        ["09"],
        ["09"],
        ["09"],
        ["14", "15", "16"],
        ["13"],
        ["13"],
        ["13"],
    ]
    forest_graph_dict = dict(zip(forest_keys, neighbor_lists_forest))
    with open("forest_graph.json", "w") as out_file:
        json.dump(forest_graph_dict, out_file, indent=1)

    # generate a small k-regular graph (k = 4)
    # do so by generating a cycle graph on 8 nodes and connecting
    # each node to its nearest four neighbors
    k_regular_keys = [str(x).zfill(2) for x in list(range(8))]
    neighbor_lists_k_reg = [
        [
            str(x).zfill(2)
            for x in [
                key_index - 2 + (key_index - 2 < 0) * 8,
                key_index - 1 + (key_index - 1 < 0) * 8,
                key_index + 1 - (key_index + 1 > 7) * 8,
                key_index + 2 - (key_index + 2 > 7) * 8,
            ]
        ]
        for key_index in range(len(k_regular_keys))
    ]
    kreg_graph_dict = dict(zip(k_regular_keys, neighbor_lists_k_reg))
    with open("kreg_graph.json", "w") as out_file:
        json.dump(kreg_graph_dict, out_file, indent=1)


if __name__ == "__main__":
    create_default_graphs()
