import pytest
import numpy as np
from typing import Any
import pandas as pd
from rag.utils.doc_store_conn import (
    DocStoreConnection,
    MatchTextExpr,
    MatchDenseExpr,
    OrderByExpr
)
from rag.utils.postgres_conn import PostgresConnection
from rag.utils.infinity_conn import InfinityConnection
from rag.utils.es_conn import ESConnection
from rag.models.chunk import Chunk

TEST_IMPLEMENTATIONS = [
    ("postgres", PostgresConnection),
    ("infinity", InfinityConnection),
    ("elasticsearch", ESConnection)
]

@pytest.fixture(params=TEST_IMPLEMENTATIONS, ids=[impl[0] for impl in TEST_IMPLEMENTATIONS])
def doc_store(request) -> DocStoreConnection:
    """Fixture that provides each DocStoreConnection implementation"""
    impl_name, impl_class = request.param
    conn = impl_class()
    yield conn

@pytest.fixture
def test_chunks():
    """Create test chunk data"""
    chunks = []
    for i in range(5):
        chunk = Chunk(
            id=f"test_{i}",
            doc_id=f"doc_{i}",
            kb_id="test_kb",
            content_ltks=f"test content {i}",
            important_kwd=[f"keyword_{i}", f"important_{i}"],
            question_kwd=[f"question_{i}"],
            position_int=[[i, i+1, i+2, i+3, i+4]],
            vector_embeddings={
                "q_4_vec": [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
            }
        )
        chunks.append(chunk)
    return chunks

class TestDocStore:
    """Tests for DocStoreConnection implementations"""

    def test_index_operations(self, doc_store):
        """Test basic index operations"""
        # Test index creation
        doc_store.createIdx("test_index", "test_kb", 4)
        assert doc_store.indexExist("test_index", "test_kb")
        
        # Test index deletion
        doc_store.deleteIdx("test_index", "test_kb")
        assert not doc_store.indexExist("test_index", "test_kb")

    def test_crud_operations(self, doc_store, test_chunks):
        """Test CRUD operations"""
        # Setup
        doc_store.createIdx("test_index", "test_kb", 4)
        docs = [chunk.to_dict() for chunk in test_chunks]
        
        # Test insert
        doc_store.insert(docs, "test_index", "test_kb")
        
        # Test get
        for chunk in test_chunks:
            result = doc_store.get(chunk.id, "test_index", ["test_kb"])
            assert result is not None
            assert result["id"] == chunk.id
            assert result["doc_id"] == chunk.doc_id
        
        # Test update
        update_values = {
            "content_ltks": "updated content",
            "important_kwd": ["new_keyword"]
        }
        doc_store.update(
            condition={"id": "test_0"},
            newValue=update_values,
            indexName="test_index",
            knowledgebaseId="test_kb"
        )
        
        # Verify update
        result = doc_store.get("test_0", "test_index", ["test_kb"])
        assert result["content_ltks"] == "updated content"
        assert "new_keyword" in result["important_kwd"].split("###")
        
        # Test delete
        doc_store.delete({"id": "test_0"}, "test_index", "test_kb")
        assert doc_store.get("test_0", "test_index", ["test_kb"]) is None

    def test_search_operations(self, doc_store, test_chunks):
        """Test search operations"""
        # Setup
        doc_store.createIdx("test_index", "test_kb", 4)
        docs = [chunk.to_dict() for chunk in test_chunks]
        doc_store.insert(docs, "test_index", "test_kb")

        # Test text search
        text_result = doc_store.search(
            selectFields=["id", "content_ltks"],
            highlightFields=["content_ltks"],
            condition={},
            matchExprs=[MatchTextExpr(
                fields=["content_ltks"],
                matching_text="test content",
                topn=10
            )],
            orderBy=OrderByExpr(),
            offset=0,
            limit=10,
            indexNames="test_index",
            knowledgebaseIds=["test_kb"]
        )
        assert isinstance(text_result[0], pd.DataFrame)
        assert text_result[1] > 0

        # Test vector search
        vector_result = doc_store.search(
            selectFields=["id", "q_4_vec"],
            highlightFields=[],
            condition={},
            matchExprs=[MatchDenseExpr(
                vector_column_name="q_4_vec",
                embedding_data=np.array([0.1, 0.2, 0.3, 0.4]),
                embedding_data_type="float",
                distance_type="cosine",
                topn=3,
                extra_options={"threshold": 0.8}
            )],
            orderBy=OrderByExpr(),
            offset=0,
            limit=3,
            indexNames="test_index",
            knowledgebaseIds=["test_kb"]
        )
        assert isinstance(vector_result[0], pd.DataFrame)
        assert vector_result[1] > 0

    def test_helper_methods(self, doc_store, test_chunks):
        """Test helper methods"""
        # Setup
        doc_store.createIdx("test_index", "test_kb", 4)
        docs = [chunk.to_dict() for chunk in test_chunks]
        doc_store.insert(docs, "test_index", "test_kb")
        
        result = doc_store.search(
            selectFields=["id", "content_ltks"],
            highlightFields=["content_ltks"],
            condition={},
            matchExprs=[],
            orderBy=OrderByExpr(),
            offset=0,
            limit=10,
            indexNames="test_index",
            knowledgebaseIds=["test_kb"]
        )
        
        # Test getTotal
        total = doc_store.getTotal(result)
        assert total > 0
        
        # Test getChunkIds
        chunk_ids = doc_store.getChunkIds(result)
        assert len(chunk_ids) > 0
        assert all(isinstance(id, str) for id in chunk_ids)
        
        # Test getFields
        fields = doc_store.getFields(result, ["content_ltks"])
        assert len(fields) > 0
        assert all("content_ltks" in doc for doc in fields.values())
        
        # Test getHighlight
        highlights = doc_store.getHighlight(result, ["test"], "content_ltks")
        assert isinstance(highlights, dict)

    def teardown_method(self, method):
        """Clean up after each test"""
        for _, impl_class in TEST_IMPLEMENTATIONS:
            conn = impl_class()
            try:
                conn.deleteIdx("test_index", "test_kb")
            except:
                pass
