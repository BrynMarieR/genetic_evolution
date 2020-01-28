from typing import Tuple, List, Dict, Any


class Graph(object):
    def __init__(self, graph: Any = None) -> None:
        if graph is None:
            graph = {}
        self.__graph: Dict[str, List[str]] = graph

    def get_vertices(self) -> List[str]:
        return list(self.__graph.keys())

    def get_edges(self) -> List[Tuple[str, str]]:
        return list(self.__calculate_edges())

    def __calculate_edges(self) -> List[Tuple[str, str]]:
        edges_list: List[Tuple[str, str]] = []
        for vertex in list(self.__graph.keys()):
            for neighbor in self.__graph[vertex]:
                if (vertex, neighbor) not in edges_list:
                    edges_list.append((vertex, neighbor))
        return edges_list

    def add_vertex(self, vertex: str) -> None:
        if vertex not in self.__graph.keys():
            self.__graph[vertex] = []
        else:
            print("Vertex already in graph!")

    def add_edge(self, edge: Tuple[str, str]) -> None:
        if edge[0] not in self.__graph.keys():
            self.__graph[edge[0]] = [edge[1]]
        else:
            self.__graph[edge[0]].append(edge[1])

    def add_vertices(self, vertices: List[str]) -> None:
        for vertex in vertices:
            self.add_vertex(vertex)

    def add_edges(self, edges: List[Tuple[str, str]]) -> None:
        for edge in edges:
            self.add_edge(edge)

    def __str__(self) -> str:
        return str(self.__graph)
