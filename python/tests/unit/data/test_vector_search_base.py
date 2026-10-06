# Copyright (c) Microsoft. All rights reserved.


import ast
from collections.abc import Callable

import pytest

from semantic_kernel.data.vector import VectorSearch, VectorSearchOptions, VectorSearchProtocol


async def test_search(vector_store_record_collection: VectorSearch):
    assert isinstance(vector_store_record_collection, VectorSearchProtocol)
    record = {"id": "test_id", "content": "test_content", "vector": [1.0, 2.0, 3.0]}
    await vector_store_record_collection.upsert(record)
    results = await vector_store_record_collection.search(vector=[1.0, 2.0, 3.0])
    records = [rec async for rec in results.results]
    assert records[0].record == record


@pytest.mark.parametrize("include_vectors", [True, False])
async def test_get_vector_search_results(vector_store_record_collection: VectorSearch, include_vectors: bool):
    options = VectorSearchOptions(include_vectors=include_vectors)
    results = [{"id": "test_id", "content": "test_content", "vector": [1.0, 2.0, 3.0]}]
    async for result in vector_store_record_collection._get_vector_search_results_from_results(
        results=results, options=options
    ):
        assert result.record == results[0] if include_vectors else {"id": "test_id", "content": "test_content"}
        break


@pytest.fixture
def parsed_nodes(DictVectorStoreRecordCollection, definition) -> Callable[[str], list[ast.AST]]:
    """Build filters with a collection whose parser returns the nodes it was given."""

    class RecordingCollection(DictVectorStoreRecordCollection):
        def _lambda_parser(self, node: ast.AST) -> ast.AST:
            return node

    collection = RecordingCollection(collection_name="test", record_type=dict, definition=definition)

    def build(filter_: str) -> list[ast.AST]:
        built = collection._build_filter(filter_)
        return built if isinstance(built, list) else [built]

    return build


@pytest.mark.parametrize(
    "literal, expected",
    [("-5", -5), ("-2.5", -2.5), ("--5", 5), ("-0", 0)],
)
def test_build_filter_folds_negative_number(parsed_nodes, literal: str, expected: int | float):
    [node] = parsed_nodes(f"lambda x: x.content > {literal}")

    assert isinstance(node, ast.Compare)
    constant = node.comparators[0]
    assert isinstance(constant, ast.Constant)
    assert constant.value == expected
    assert type(constant.value) is type(expected)


def test_build_filter_folds_negative_number_in_chained_comparison(parsed_nodes):
    [node] = parsed_nodes("lambda x: -10 < x.content < -1")

    assert isinstance(node, ast.Compare)
    assert isinstance(node.left, ast.Constant) and node.left.value == -10
    assert isinstance(node.comparators[1], ast.Constant) and node.comparators[1].value == -1


@pytest.mark.parametrize("expression", ["+5", "-True", "-x.content", "~5", "-'a'"])
def test_build_filter_keeps_other_unary_operators(parsed_nodes, expression: str):
    [node] = parsed_nodes(f"lambda x: x.content == {expression}")

    assert isinstance(node, ast.Compare)
    assert isinstance(node.comparators[0], ast.UnaryOp)
