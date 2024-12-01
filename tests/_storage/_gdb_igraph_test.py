# type: ignore

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from fast_graphrag._storage._gdb_igraph import IGraphStorage, IGraphStorageConfig
from fast_graphrag._types import TEntity, TRelation


class TestIGraphStorage(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = IGraphStorageConfig(node_cls=TEntity, edge_cls=TRelation)
        self.storage = IGraphStorage(config=self.config)
        self.storage._graph = MagicMock()

    async def test_node_count(self):
        self.storage._graph.vcount.return_value = 10
        count = await self.storage.node_count()
        self.assertEqual(count, 10)

    async def test_edge_count(self):
        self.storage._graph.ecount.return_value = 20
        count = await self.storage.edge_count()
        self.assertEqual(count, 20)

    async def test_get_node(self):
        node = MagicMock()
        node.name = "node1"
        node.attributes.return_value = {
            "name": "foo", 
            "description": "value",
            "type": "some_entity_type"
        }
        self.storage._graph.vs.find.return_value = node

        result = await self.storage.get_node("node1")
        self.assertEqual(result, (TEntity(**node.attributes()), node.index))

    async def test_get_node_not_found(self):
        self.storage._graph.vs.find.side_effect = ValueError
        result = await self.storage.get_node("node1")
        self.assertEqual(result, (None, None))

    async def test_get_edges(self):
        self.storage.get_edge_indices = AsyncMock(return_value=[0, 1])
        self.storage.get_edge_by_index = AsyncMock(
            side_effect=[TRelation(source="node1", target="node2", description="txt", keywords=["Keyword1", "Keyword2"]), None]
        )

        edges = await self.storage.get_edges("node1", "node2")
        self.assertEqual(edges, [(TRelation(source="node1", target="node2", description="txt", keywords=["Keyword1", "Keyword2"]), 0)])

    async def test_get_edge_indices(self):
        self.storage._graph.vs.find.side_effect = lambda name: MagicMock(index=name)
        self.storage._graph.es.select.return_value = [MagicMock(index=0), MagicMock(index=1)]

        indices = await self.storage.get_edge_indices("node1", "node2")
        self.assertEqual(list(indices), [0, 1])

    async def test_get_node_by_index(self):
        node = MagicMock()
        node.attributes.return_value = {
            "name": "foo", 
            "description": "value",
            "type": "some_entity_type"
        }
        self.storage._graph.vs.__getitem__.return_value = node
        self.storage._graph.vcount.return_value = 1

        result = await self.storage.get_node_by_index(0)
        self.assertEqual(result, TEntity(**node.attributes()))

    async def test_get_edge_by_index(self):
        edge = MagicMock()
        edge.source = "node0"
        edge.target = "node1"
        edge.attributes.return_value = {"description": "value", "keywords": ["Keyword1", "Keyword2"]}
        self.storage._graph.es.__getitem__.return_value = edge
        self.storage._graph.vs.__getitem__.side_effect = lambda idx: {"name": idx}
        self.storage._graph.ecount.return_value = 1

        result = await self.storage.get_edge_by_index(0)
        self.assertEqual(result, TRelation(source="node0", target="node1", **edge.attributes()))

    async def test_upsert_node(self):
        node = TEntity(name="node1", description="value", type="node type")
        self.storage._graph.vcount.return_value = 1
        self.storage._graph.vs.__getitem__.return_value = MagicMock(index=0)

        index = await self.storage.upsert_node(node, 0)
        self.assertEqual(index, 0)

    async def test_upsert_edge(self):
        edge = TRelation(
            source="node1",
            target="node2",
            description="desc",
            chunks=[],
            keywords=[] 
        )
        self.storage._graph.ecount.return_value = 1
        self.storage._graph.es.__getitem__.return_value = MagicMock(index=0)

        index = await self.storage.upsert_edge(edge, 0)
        self.assertEqual(index, 0)

    async def test_delete_edges_by_index(self):
        self.storage._graph.delete_edges = MagicMock()
        indices = [0, 1]
        await self.storage.delete_edges_by_index(indices)
        self.storage._graph.delete_edges.assert_called_with(indices)

    @patch("fast_graphrag._storage._gdb_igraph.logger")
    async def test_score_nodes_empty_graph(self, mock_logger):
        self.storage._graph.vcount.return_value = 0
        scores = await self.storage.score_nodes(None)
        self.assertEqual(scores.shape, (1, 0))
        mock_logger.info.assert_called_with("Trying to score nodes in an empty graph.")

    async def test_score_nodes(self):
        self.storage._graph.vcount.return_value = 3
        self.storage._graph.personalized_pagerank.return_value = [0.1, 0.2, 0.7]

        scores = await self.storage.score_nodes(None)
        self.assertTrue(np.array_equal(scores.toarray(), np.array([[0.1, 0.2, 0.7]], dtype=np.float32)))

    async def test_get_entities_to_relationships_map_empty_graph(self):
        self.storage._graph.vs = []
        result = await self.storage.get_entities_to_relationships_map()
        self.assertEqual(result.shape, (0, 0))

    @patch("fast_graphrag._storage._gdb_igraph.csr_from_indices_list")
    async def test_get_entities_to_relationships_map(self, mock_csr_from_indices_list):
        self.storage._graph.vs = [MagicMock(incident=lambda: [MagicMock(index=0), MagicMock(index=1)])]
        self.storage.node_count = AsyncMock(return_value=1)
        self.storage.edge_count = AsyncMock(return_value=2)

        await self.storage.get_entities_to_relationships_map()
        mock_csr_from_indices_list.assert_called_with([[0, 1]], shape=(1, 2))

    async def test_get_relationships_attrs_empty_graph(self):
        self.storage._graph.es = []
        result = await self.storage.get_relationships_attrs("key")
        self.assertEqual(result, [])

    async def test_get_relationships_attrs(self):
        self.storage._graph.es.__getitem__.return_value = [[1, 2], [3, 4]]
        self.storage._graph.es.__len__.return_value = 2
        result = await self.storage.get_relationships_attrs("key")
        self.assertEqual(result, [[1, 2], [3, 4]])

    @patch("igraph.Graph.Read_Picklez")
    @patch("fast_graphrag._storage._gdb_igraph.logger")
    async def test_insert_start_with_existing_file(self, mock_logger, mock_read_picklez):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.return_value = "dummy_path"

        await self.storage._insert_start()

        mock_read_picklez.assert_called_with("dummy_path")
        mock_logger.debug.assert_called_with("Loaded graph storage 'dummy_path'.")

    @patch("igraph.Graph")
    @patch("fast_graphrag._storage._gdb_igraph.logger")
    async def test_insert_start_with_no_file(self, mock_logger, mock_graph):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.return_value = None

        await self.storage._insert_start()

        mock_graph.assert_called_with(directed=False)
        mock_logger.info.assert_called_with("No data file found for graph storage 'None'. Loading empty graph.")

    @patch("igraph.Graph")
    @patch("fast_graphrag._storage._gdb_igraph.logger")
    async def test_insert_start_with_no_namespace(self, mock_logger, mock_graph):
        self.storage.namespace = None

        await self.storage._insert_start()

        mock_graph.assert_called_with(directed=False)
        mock_logger.debug.assert_called_with("Creating new volatile graphdb storage.")

    @patch("igraph.Graph.write_picklez")
    @patch("fast_graphrag._storage._gdb_igraph.logger")
    async def test_insert_done(self, mock_logger, mock_write_picklez):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_save_path.return_value = "dummy_path"

        await self.storage._insert_done()

        mock_write_picklez.assert_called_with(self.storage._graph, "dummy_path")

    @patch("igraph.Graph.Read_Picklez")
    @patch("fast_graphrag._storage._gdb_igraph.logger")
    async def test_query_start_with_existing_file(self, mock_logger, mock_read_picklez):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.return_value = "dummy_path"

        await self.storage._query_start()

        mock_read_picklez.assert_called_with("dummy_path")
        mock_logger.debug.assert_called_with("Loaded graph storage 'dummy_path'.")

    @patch("igraph.Graph")
    @patch("fast_graphrag._storage._gdb_igraph.logger")
    async def test_query_start_with_no_file(self, mock_logger, mock_graph):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.return_value = None

        await self.storage._query_start()

        mock_graph.assert_called_with(directed=False)
        mock_logger.warning.assert_called_with(
            "No data file found for graph storage 'None'. Loading empty graph."
        )


if __name__ == "__main__":
    unittest.main()
