# Copyright (c) Microsoft. All rights reserved.

from pytest import fixture

from semantic_kernel.connectors.qdrant import QdrantCollection
from semantic_kernel.data.vector import VectorStoreCollectionDefinition, VectorStoreField

# These tests run against the in-process Qdrant that ships with qdrant-client, so they exercise the real client API.

RECORDS = [
    {"id": 1, "temperature": 1, "vector": [1.0, 0.2]},
    {"id": 2, "temperature": 5, "vector": [0.0, 1.0]},
    {"id": 3, "temperature": 9, "vector": [1.0, 1.0]},
]


@fixture(params=[True, False], ids=["named_vectors", "unnamed_vector"])
async def local_collection(request):
    definition = VectorStoreCollectionDefinition(
        fields=[
            VectorStoreField("key", name="id", type="int"),
            VectorStoreField("data", name="temperature", type="int", is_indexed=True),
            VectorStoreField("vector", name="vector", dimensions=2, type="float"),
        ]
    )
    async with QdrantCollection(
        record_type=dict,
        definition=definition,
        collection_name="local_test",
        location=":memory:",
        named_vectors=request.param,
    ) as collection:
        await collection.ensure_collection_exists()
        await collection.upsert(RECORDS)
        yield collection


async def _ids(results) -> list[int]:
    return [result.record["id"] async for result in results.results]


async def test_local_vector_search(local_collection):
    assert await _ids(await local_collection.search(vector=[1.0, 1.0], top=3)) == [3, 1, 2]


async def test_local_vector_search_top_and_skip(local_collection):
    assert await _ids(await local_collection.search(vector=[1.0, 1.0], top=1)) == [3]
    assert await _ids(await local_collection.search(vector=[1.0, 1.0], top=1, skip=1)) == [1]


async def test_local_vector_search_single_filter(local_collection):
    results = await local_collection.search(vector=[1.0, 1.0], filter="lambda x: x.temperature > 1")
    assert await _ids(results) == [3, 2]


async def test_local_vector_search_multiple_filters(local_collection):
    results = await local_collection.search(
        vector=[1.0, 1.0], filter=["lambda x: x.temperature > 1", "lambda x: x.temperature < 9"]
    )
    assert await _ids(results) == [2]
