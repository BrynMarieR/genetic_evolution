from typing import Tuple, List, Any


class Graph(object):
    def __init__(self, graph: Any) -> None:
        if graph is None:
            graph = {}
        self.graph = graph

    def get_vertices(self) -> List[str]:
        return list(self.graph.keys())

    def get_edges(self) -> List[Tuple[str, str]]:
        return list(self.__calculate_edges())

    def __calculate_edges(self) -> List[Tuple[str, str]]:
        edges_list: List[Tuple[str, str]] = []
        for vertex in list(self.graph.keys()):
            for neighbor in self.graph[vertex]:
                if (vertex, neighbor) not in edges_list:
                    edges_list.append((vertex, neighbor))
        return edges_list

    def add_vertex(self, vertex: str) -> None:
        if vertex not in self.graph.keys():
            self.graph[vertex] = []
        else:
            print("Vertex already in graph!")

    def add_edge(self, edge: Tuple[str, str]) -> None:
        if edge[0] not in self.graph.keys():
            self.graph[edge[0]] = [edge[1]]
        else:
            self.graph[edge[0]].append(edge[1])

    def add_vertices(self, vertices: List[str]) -> None:
        for vertex in vertices:
            self.add_vertex(vertex)

    def add_edges(self, edges: List[Tuple[str, str]]) -> None:
        for edge in edges:
            self.add_edge(edge)

    def __str__(self) -> str:
        return str(self.graph)

    def __len__(self) -> int:
        return len(self.graph)
