import gzip
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Generic, Iterable, List, Mapping, Optional, Tuple, Type, Union

import igraph as ig  # type: ignore
import numpy as np
from scipy.sparse import csr_matrix

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._types import GTEdge, GTId, GTNode, TIndex
from fast_graphrag._utils import csr_from_indices_list, logger

from ._base import BaseGraphStorage


@dataclass
class IGraphStorageConfig(Generic[GTNode, GTEdge]):
    node_cls: Type[GTNode] = field()
    edge_cls: Type[GTEdge] = field()
    ppr_damping: float = field(default=0.85)


@dataclass
class IGraphStorage(BaseGraphStorage[GTNode, GTEdge, GTId]):
    RESOURCE_NAME = "igraph_data.pklz"
    config: IGraphStorageConfig[GTNode, GTEdge] = field()
    _graph: Optional[ig.Graph] = field(init=False, default=None)  # type: ignore

    async def save_graphml(self, path: str) -> None:
        if self._graph is not None:  # type: ignore
            ig.Graph.write_graphmlz(self._graph, path + ".gz")  # type: ignore

            with gzip.open(path + ".gz", "rb") as f:
                file_content = f.read()
            with open(path, "wb") as f:
                f.write(file_content)
            os.remove(path + ".gz")

    async def node_count(self) -> int:
        return self._graph.vcount()  # type: ignore

    async def edge_count(self) -> int:
        return self._graph.ecount()  # type: ignore

    async def get_node(self, node: Union[GTNode, GTId]) -> Union[Tuple[GTNode, TIndex], Tuple[None, None]]:
        if isinstance(node, self.config.node_cls):
            node_id = node.name
        else:
            node_id = node

        try:
            vertex = self._graph.vs.find(name=node_id)  # type: ignore
        except ValueError:
            vertex = None

        return (self.config.node_cls(**vertex.attributes()), vertex.index) if vertex else (None, None)  # type: ignore

    async def get_edges(
        self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]
    ) -> Iterable[Tuple[GTEdge, TIndex]]:
        indices = await self.get_edge_indices(source_node, target_node)
        edges: List[Tuple[GTEdge, TIndex]] = []
        for index in indices:
            edge = await self.get_edge_by_index(index)
            if edge:
                edges.append((edge, index))
        return edges

    async def get_edge_indices(
        self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]
    ) -> Iterable[TIndex]:
        if type(source_node) is TIndex:
            source_node = self._graph.vs.find(name=source_node).index  # type: ignore
        if type(target_node) is TIndex:
            target_node = self._graph.vs.find(name=target_node).index  # type: ignore
        edges = self._graph.es.select(_source=source_node, _target=target_node)  # type: ignore

        return (edge.index for edge in edges)  # type: ignore

    async def get_node_by_index(self, index: TIndex) -> Union[GTNode, None]:
        node = self._graph.vs[index] if index < self._graph.vcount() else None  # type: ignore
        return self.config.node_cls(**node.attributes()) if index < self._graph.vcount() else None  # type: ignore

    async def get_edge_by_index(self, index: TIndex) -> Union[GTEdge, None]:
        edge = self._graph.es[index] if index < self._graph.ecount() else None  # type: ignore
        return (
            self.config.edge_cls(
                source=self._graph.vs[edge.source]["name"],  # type: ignore
                target=self._graph.vs[edge.target]["name"],  # type: ignore
                **edge.attributes(),  # type: ignore
            )
            if edge
            else None
        )

    async def upsert_node(self, node: GTNode, node_index: Union[TIndex, None]) -> TIndex:
        if node_index is not None:
            if node_index >= self._graph.vcount():  # type: ignore
                logger.error(
                    f"Trying to update node with index {node_index} but graph has only {self._graph.vcount()} nodes."  # type: ignore
                )
                raise ValueError(f"Index {node_index} is out of bounds")
            already_node = self._graph.vs[node_index]  # type: ignore
            already_node.update_attributes(**asdict(node))  # type: ignore

            return already_node.index  # type: ignore
        else:
            return self._graph.add_vertex(**asdict(node)).index  # type: ignore

    async def upsert_edge(self, edge: GTEdge, edge_index: Union[TIndex, None]) -> TIndex:
        if edge_index is not None:
            if edge_index >= self._graph.ecount():  # type: ignore
                logger.error(
                    f"Trying to update edge with index {edge_index} but graph has only {self._graph.ecount()} edges."  # type: ignore
                )
                raise ValueError(f"Index {edge_index} is out of bounds")
            already_edge = self._graph.es[edge_index]  # type: ignore
            already_edge.update_attributes(**edge.to_attrs(edge=edge))  # type: ignore

            return already_edge.index  # type: ignore
        else:
            return self._graph.add_edge(  # type: ignore
                **asdict(edge)
            ).index  # type: ignore

    async def insert_edges(
        self,
        edges: Optional[Iterable[GTEdge]] = None,
        indices: Optional[Iterable[Tuple[TIndex, TIndex]]] = None,
        attrs: Optional[Mapping[str, Iterable[Any]]] = None,
    ) -> List[TIndex]:
        if indices is not None:
            assert edges is None, "Cannot provide both indices and edges."

            indices = list(indices)
            if len(indices) == 0:
                return []
            self._graph.add_edges(  # type: ignore
                indices,
                attributes=attrs,
            )
            # TODO: not sure if this is the best way to get the indices of the new edges
            return list(range(self._graph.ecount() - len(indices), self._graph.ecount()))  # type: ignore
        elif edges is not None:
            assert indices is None and attrs is None, "Cannot provide both indices and edges."
            edges = list(edges)
            if len(edges) == 0:
                return []
            self._graph.add_edges(  # type: ignore
                ((edge.source, edge.target) for edge in edges),
                attributes=type(edges[0]).to_attrs(edges=edges),
            )
            # TODO: not sure if this is the best way to get the indices of the new edges
            return list(range(self._graph.ecount() - len(edges), self._graph.ecount()))  # type: ignore
        else:
            return []

    async def are_neighbours(self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]) -> bool:
        return self._graph.get_eid(source_node, target_node, directed=False, error=False) != -1  # type: ignore

    async def delete_edges_by_index(self, indices: Iterable[TIndex]) -> None:
        self._graph.delete_edges(indices)  # type: ignore

    async def score_nodes(self, initial_weights: Optional[csr_matrix]) -> csr_matrix:
        if self._graph.vcount() == 0:  # type: ignore
            logger.info("Trying to score nodes in an empty graph.")
            return csr_matrix((1, 0))

        reset_prob = initial_weights.toarray().flatten() if initial_weights is not None else None

        ppr_scores = self._graph.personalized_pagerank(  # type: ignore
            damping=self.config.ppr_damping, directed=False, reset=reset_prob
        )
        ppr_scores = np.array(ppr_scores, dtype=np.float32)  # type: ignore

        return csr_matrix(
            ppr_scores.reshape(1, -1)  # type: ignore
        )

    async def get_entities_to_relationships_map(self) -> csr_matrix:
        if len(self._graph.vs) == 0:  # type: ignore
            return csr_matrix((0, 0))

        return csr_from_indices_list(
            [
                [edge.index for edge in vertex.incident()]  # type: ignore
                for vertex in self._graph.vs  # type: ignore
            ],
            shape=(await self.node_count(), await self.edge_count()),
        )

    async def get_relationships_attrs(self, key: str) -> List[List[Any]]:
        if len(self._graph.es) == 0:  # type: ignore
            return []

        lists_of_attrs: List[List[TIndex]] = []
        for attr in self._graph.es[key]:  # type: ignore
            lists_of_attrs.append(list(attr))  # type: ignore

        return lists_of_attrs

    async def _insert_start(self):
        if self.namespace:
            graph_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)

            if graph_file_name:
                try:
                    self._graph = ig.Graph.Read_Picklez(graph_file_name)  # type: ignore
                    logger.debug(f"Loaded graph storage '{graph_file_name}'.")
                except Exception as e:
                    t = f"Error loading graph from {graph_file_name}: {e}"
                    logger.error(t)
                    raise InvalidStorageError(t) from e
            else:
                logger.info(f"No data file found for graph storage '{graph_file_name}'. Loading empty graph.")
                self._graph = ig.Graph(directed=False)
        else:
            self._graph = ig.Graph(directed=False)
            logger.debug("Creating new volatile graphdb storage.")

    async def _insert_done(self):
        if self.namespace:
            graph_file_name = self.namespace.get_save_path(self.RESOURCE_NAME)
            try:
                ig.Graph.write_picklez(self._graph, graph_file_name)  # type: ignore
            except Exception as e:
                t = f"Error saving graph to {graph_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e

    async def _query_start(self):
        assert self.namespace, "Loading a graph requires a namespace."
        graph_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)
        if graph_file_name:
            try:
                self._graph = ig.Graph.Read_Picklez(graph_file_name)  # type: ignore
                logger.debug(f"Loaded graph storage '{graph_file_name}'.")
            except Exception as e:
                t = f"Error loading graph from '{graph_file_name}': {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e
        else:
            logger.warning(f"No data file found for graph storage '{graph_file_name}'. Loading empty graph.")
            self._graph = ig.Graph(directed=False)

    async def _query_done(self):
        pass
