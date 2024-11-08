# type: ignore
import copy
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

from fast_graphrag._llm._llm_openai import BaseLLMService
from fast_graphrag._policies._graph_upsert import (
    DefaultEdgeUpsertPolicy,
    # DefaultGraphUpsertPolicy,
    DefaultNodeUpsertPolicy,
    EdgeUpsertPolicy_UpsertIfValidNodes,
    EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM,
    NodeUpsertPolicy_SummarizeDescription,
)
from fast_graphrag._storage._base import BaseGraphStorage


class TestNodeUpsertPolicy_SummarizeDescription(unittest.IsolatedAsyncioTestCase):  # noqa: N801
    async def test_call_same_node_summarize(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        node1 = MagicMock()
        node1.name = "node1"
        node1.description = "This is a lengthy random description."
        node2 = MagicMock()
        node2.name = "node1"
        node2.description = "This is a lengthy random description that is being used to test the summarization."
        source_nodes = [node1, node2]

        # Mock methods
        llm.send_message.return_value = ("This is a summary.", None)
        target.get_node = AsyncMock(return_value=(None, None))
        target.upsert_node.side_effect = lambda node, node_index: node_index or 0

        # Create policy instance
        policy = NodeUpsertPolicy_SummarizeDescription(
            config=NodeUpsertPolicy_SummarizeDescription.Config(
                max_node_description_tokens=len(node1.description) + 4,
            )
        )

        # Call the method
        _, upserted_nodes = await policy(llm, target, source_nodes)
        self.assertEqual(upserted_nodes[0][1].name, "node1")
        self.assertEqual(upserted_nodes[0][1].description, "This is a summary.")

    async def test_call_same_node_no_summarize(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        node1 = MagicMock()
        node1.name = "node1"
        node1.description = "This is a short random description 1."
        node2 = MagicMock()
        node2.name = "node1"
        node2.description = "This is a short random description 2."
        source_nodes = [node1, node2]

        # Mock methods
        llm.send_message.return_value = ("This is a summary.", None)
        target.get_node.side_effect = [(None, None), (node1, 0)]
        target.upsert_node.side_effect = lambda node, node_index: node_index or 0

        # Create policy instance
        policy = NodeUpsertPolicy_SummarizeDescription(
            config=NodeUpsertPolicy_SummarizeDescription.Config(
                max_node_description_tokens=(len(node1.description) * 2)
            )
        )

        # Call the method
        _, upserted_nodes = await policy(llm, target, source_nodes)

        self.assertEqual(
            upserted_nodes[0][1].description,
            "This is a short random description 1.\nThis is a short random description 2.",
        )

        # Assertions
        llm.send_message.assert_not_called()

    async def test_call_two_nodes(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        node1 = MagicMock()
        node1.name = "node1"
        node1.description = "Description for node1."
        node2 = MagicMock()
        node2.name = "node2"
        node2.description = "Description for node2."
        source_nodes = [node1, node2]

        # Mock methods
        llm.send_message.return_value = ("This is a summary.", None)
        target.get_node.side_effect = [(None, None), (None, None)]
        target.upsert_node.side_effect = lambda node, node_index: node_index or 0

        # Create policy instance
        policy = NodeUpsertPolicy_SummarizeDescription(
            config=NodeUpsertPolicy_SummarizeDescription.Config(
                max_node_description_tokens=len(node1.description),
            )
        )

        # Call the method
        _, upserted_nodes = await policy(llm, target, source_nodes)

        self.assertEqual(upserted_nodes[0][1].description, "Description for node1.")
        self.assertEqual(upserted_nodes[1][1].description, "Description for node2.")


class TestEdgeUpsertPolicy_UpsertIfValidNodes(unittest.IsolatedAsyncioTestCase):  # noqa: N801
    async def test_call(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        edge1 = MagicMock()
        edge1.source = "source1"
        edge1.target = "target1"
        edge2 = MagicMock()
        edge2.source = "source1"
        edge2.target = "target2"
        source_edges = [edge1, edge2]

        # Mock methods
        target.get_node.side_effect = lambda x: (MagicMock(), None) if x in ["source1", "target1"] else (None, None)
        target.upsert_edge = AsyncMock()

        # Create policy instance
        policy = EdgeUpsertPolicy_UpsertIfValidNodes(config=EdgeUpsertPolicy_UpsertIfValidNodes.Config())

        # Call the method
        await policy(llm, target, source_edges)

        # Assertions
        target.get_node.assert_has_calls([call("source1"), call("target1"), call("source1"), call("target2")])
        target.upsert_edge.assert_called_once_with(edge=edge1, edge_index=None)


class TestDefaultNodeUpsertPolicy(unittest.IsolatedAsyncioTestCase):  # noqa: N801
    async def test_call_same_id(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        node1 = MagicMock()
        node1.name = "node1"
        node1.description = "Description for node1."
        node2 = MagicMock()
        node2.name = "node1"
        node2.description = "Description for node2."
        source_nodes = [node1, node2]

        # Mock methods
        target.get_node = AsyncMock()
        target.get_node.side_effect = [(None, None), (node1, 0)]
        target.upsert_node = AsyncMock()
        target.upsert_node.side_effect = lambda node, node_index: node_index or 0

        # Create policy instance
        policy = DefaultNodeUpsertPolicy(config=None)

        # Call the method
        _, upserted = await policy(llm, target, source_nodes)
        upserted = list(upserted)

        self.assertEqual(len(upserted), 1)
        self.assertEqual(upserted[0][1].description, "Description for node2.")

        # Assertions
        target.get_node.assert_has_calls([call(node1), call(node2)])
        target.upsert_node.assert_has_calls([call(node=node1, node_index=None), call(node=node2, node_index=0)])

    async def test_call_different_id(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        node1 = MagicMock()
        node1.name = "node1"
        node1.description = "Description for node1."
        node2 = MagicMock()
        node2.name = "node2"
        node2.description = "Description for node2."
        source_nodes = [node1, node2]

        # Mock methods
        target.get_node = AsyncMock()
        target.get_node.side_effect = [(None, None), (None, None)]
        target.upsert_node = AsyncMock()
        target.upsert_node.side_effect = [0, 1]  # type: ignore

        # Create policy instance
        policy = DefaultNodeUpsertPolicy(config=None)

        # Call the method
        _, upserted = await policy(llm, target, source_nodes)
        upserted = list(upserted)
        self.assertEqual(len(upserted), 2)

        # Assertions
        target.get_node.assert_has_calls([call(node1), call(node2)])
        target.upsert_node.assert_has_calls([call(node=node1, node_index=None), call(node=node2, node_index=None)])


class TestEdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM(unittest.IsolatedAsyncioTestCase):  # noqa: N801
    async def asyncSetUp(self):
        self.mock_llm = AsyncMock(spec=BaseLLMService)
        self.mock_target = AsyncMock(spec=BaseGraphStorage)

    async def test_call_edges_below_threshold(self):
        edge_upsert_policy = EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM(
            EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM.Config(edge_merge_threshold=1)
        )
        sources = ["node1", "node2", "node3"]
        targets = ["node4", "node5", "node6"]

        edges = [
            MagicMock(source=source, target=target, description=source + target)
            for source in sources
            for target in targets
        ]
        self.mock_target.get_edges = AsyncMock(return_value=[])

        insert_index = 0

        def _upsert_edge(edge, edge_index):
            nonlocal insert_index
            insert_index += 1
            return insert_index

        self.mock_target.upsert_edge = AsyncMock(side_effect=_upsert_edge)
        self.mock_target.delete_edges_by_index = AsyncMock()

        target, upserted_edges = await edge_upsert_policy(self.mock_llm, self.mock_target, edges)
        self.assertEqual(len(list(upserted_edges)), 9)
        self.assertEqual(len(list(self.mock_target.delete_edges_by_index.call_args[0][0])), 0)
        self.assertEqual(insert_index, 9)
        self.mock_target.delete_edges_by_index.assert_called_once()

    @patch("fast_graphrag._policies._graph_upsert.format_and_send_prompt", new_callable=AsyncMock)
    async def test_call_edges_above_threshold(self, mock_format_and_send_prompt):
        edge_upsert_policy = EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM(
            EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM.Config(edge_merge_threshold=1)
        )
        sources = ["node1", "node2"]
        targets = ["node4", "node5"]

        edges = [
            MagicMock(source=source, target=target, description=source + target)
            for source in sources
            for target in targets
        ]
        existing_edges = {
            "node1node4": [(copy.copy(edges[0]), 1)],
            "node1node5": [
                (copy.copy(edges[1]), 0),
                (copy.copy(edges[1]), 2),
                (copy.copy(edges[1]), 3),
                (copy.copy(edges[1]), 4),
            ],
        }
        edges = [copy.copy(edges[0])] + edges  # add a duplicate for first edge
        self.mock_target.get_edges = AsyncMock(
            side_effect=lambda source_node, target_node: existing_edges.get(source_node + target_node, [])
        )

        def _format_and_send_prompt(format_kwargs, **kwargs):
            if "node4" in format_kwargs["edge_list"]:
                mock_result = MagicMock()
                mock_group = MagicMock()
                mock_group.ids = [0, 1]
                mock_group.description = "Summary"
                mock_result.groups = [mock_group]
                return mock_result, None
            elif "node5" in format_kwargs["edge_list"]:
                mock_result = MagicMock()
                mock_group1 = MagicMock()
                mock_group1.ids = [1, 2, 0]
                mock_group1.description = "Summary1"
                mock_group2 = MagicMock()
                mock_group2.ids = [4, 3]
                mock_group2.description = "Summary2"
                mock_result.groups = [mock_group1, mock_group2]
                return mock_result, None

        mock_format_and_send_prompt.side_effect = _format_and_send_prompt

        insert_index = 5  # number of existing edges

        def _upsert_edge(edge, edge_index):
            if edge_index is not None:
                return edge_index
            nonlocal insert_index
            i = insert_index
            insert_index += 1
            return i

        self.mock_target.upsert_edge = AsyncMock(side_effect=_upsert_edge)
        self.mock_target.delete_edges_by_index = AsyncMock()

        target, upserted_edges = await edge_upsert_policy(self.mock_llm, self.mock_target, edges)
        upserted_edges = list(upserted_edges)
        edges = [e[1].description for e in upserted_edges]
        self.assertEqual(edges, ['Summary', 'node1node4', 'Summary1', 'Summary2', 'node2node4', 'node2node5'])
        self.assertEqual([e[0] for e in upserted_edges], [1, 5, 2, 6, 7, 8])
        self.assertEqual(set(self.mock_target.delete_edges_by_index.call_args[0][0]), {0, 3, 4})


class TestDefaultEdgeUpsertPolicy(unittest.IsolatedAsyncioTestCase):  # noqa: N801
    async def test_call(self):
        # Mock dependencies
        llm = AsyncMock(spec=BaseLLMService)
        target = AsyncMock(spec=BaseGraphStorage)
        edge = MagicMock()
        source_edges = [edge]

        # Mock methods
        target.upsert_edge = AsyncMock()

        # Create policy instance
        policy = DefaultEdgeUpsertPolicy(config=None)

        # Call the method
        await policy(llm, target, source_edges)

        # Assertions
        target.upsert_edge.assert_called_once_with(edge=edge, edge_index=None)


# class TestDefaultGraphUpsertPolicy(unittest.IsolatedAsyncioTestCase):
#     async def test_call(self):
#         # Mock dependencies
#         llm = AsyncMock(spec=BaseLLMService)
#         source = AsyncMock(spec=BaseGraphStorage)
#         target_nodes = AsyncMock(spec=BaseGraphStorage)
#         target_edges = AsyncMock(spec=BaseGraphStorage)
#         node = MagicMock()
#         edge = MagicMock()
#         source_nodes = [node]
#         source_edges = [edge]

#         # Mock methods
#         policy = DefaultGraphUpsertPolicy[TEntity, TRelation, TId](
#             config=None,
#             nodes_upsert_cls=DefaultNodeUpsertPolicy[TEntity, TId],
#             edges_upsert_cls=DefaultEdgeUpsertPolicy[TRelation, TId],
#         )
#         policy._nodes_upsert = AsyncMock(return_value=target_nodes)
#         policy._edges_upsert = AsyncMock(return_value=target_edges)

#         # Call the method
#         result = await policy(llm, source, source_nodes, source_edges)

#         # Assertions
#         policy._nodes_upsert.assert_called_once_with(llm, source, source_nodes)
#         policy._edges_upsert.assert_called_once_with(llm, target_nodes, source_edges)
#         self.assertEqual(result, target_edges)

#     async def test_call_with_default(self):
#         pass


if __name__ == "__main__":
    unittest.main()
