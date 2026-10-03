from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from application.services.chat_service import ChatService
from domain.models.retrieval_models import CompressionResult, RetrievedChunk, SourceSpan
from domain.services.compare_merge import merge_compare_contexts
from domain.services.compressor import ExtractiveCompressor
from domain.services.retrieval_service import RetrievalService
from domain.services.sqlite_structured_query import SQLiteDataSource
from domain.services.structured_query import StructuredQueryTool, parse_structured_query_intent
from infrastructure.sqlite.seed_structured_query_db import seed_db


def test_seed_and_datasource_use_only_explicit_temporary_database(tmp_path, sqlite_datasource):
    assert Path(sqlite_datasource.db_path).is_relative_to(tmp_path)
    with TemporaryDirectory(dir=tmp_path) as directory:
        path = Path(directory) / "seed.db"
        seed_db(str(path))
        rows = SQLiteDataSource(str(path)).execute_readonly("SELECT count(*) AS result FROM sales")
        assert rows[0]["result"] > 0
        assert path.exists()
    assert not path.exists()
    assert Path(sqlite_datasource.db_path).exists()


def test_tests_reject_development_database_access():
    import sqlite3
    with pytest.raises(AssertionError, match="explicit temporary"):
        sqlite3.connect("data/structured_query.db")


@pytest.mark.parametrize("query,period", [
    ("2025 Q1の売上合計", "2025-Q1"),
    ("2025年Q1の売上合計", "2025-Q1"),
    ("2026年第1四半期の売上合計", "2026-Q1"),
    ("2026 Q1 sales 合計", "2026-Q1"),
    ("2026-Q4の売上合計", "2026-Q4"),
    ("Q1の売上合計", "2025-Q1"),
])
def test_explicit_year_and_quarter_are_preserved(query, period):
    assert parse_structured_query_intent(query).filters["period"] == period


def test_unsupported_explicit_year_returns_no_data_not_2025(sqlite_datasource):
    supported = StructuredQueryTool.run("2025年第1四半期の売上合計", datasource=sqlite_datasource)
    unsupported = StructuredQueryTool.run("2026年第1四半期の売上合計", datasource=sqlite_datasource)
    assert supported.success and supported.rows[0]["result"] > 0
    assert unsupported.filters == {"period": "2026-Q1"}
    assert unsupported.rows[0]["result"] is None
    assert "データが見つかりません" in unsupported.summary
    year_only = StructuredQueryTool.run("2026年の売上合計", datasource=sqlite_datasource)
    assert not year_only.success


@pytest.mark.anyio
async def test_compare_compressed_citations_remap_to_unique_final_sources():
    a = RetrievedChunk(doc_id="a", chunk_id="a1", content="A one. A two.")
    shared = RetrievedChunk(doc_id="shared", chunk_id="s1", content="Shared.")
    b = RetrievedChunk(doc_id="b", chunk_id="b1", content="B one.")
    compressed_a = CompressionResult(
        compressed_text="[doc1-s1] A one. [doc1-s2] A two. [doc2-s1] Shared. [2]",
        source_spans=[SourceSpan("a", "a1", 0), SourceSpan("a", "a1", 1), SourceSpan("shared", "s1", 0)],
    )
    compressed_b = CompressionResult(
        compressed_text="[doc1-s1] Shared. [doc2-s1] B one. [1, 2]",
        source_spans=[SourceSpan("shared", "s1", 0), SourceSpan("b", "b1", 0)],
    )
    with patch.object(ExtractiveCompressor, "compress", AsyncMock(side_effect=[compressed_a, compressed_b])):
        result_a = await RetrievalService.prepare_context("A", [a, shared])
        result_b = await RetrievalService.prepare_context("B", [shared, b])
    packed, _, _, covered, _, _, sources = merge_compare_contexts(["A", "B"], {
        "A": {"context": result_a.context, "sources": result_a.sources, "chunks": [a, shared]},
        "B": {"context": result_b.context, "sources": result_b.sources, "chunks": [shared, b]},
    })
    assert covered
    assert "[doc" not in packed
    assert "[1] A one. [1] A two. [2] Shared. [2]" in packed
    assert "[2] Shared. [3] B one. [2, 3]" in packed
    assert [source["citation_id"] for source in sources] == [1, 2, 3]
    trace_sources, mapping = ChatService._build_trace_sources({"sources": sources})
    assert mapping == {1: "a#a1", 2: "shared#s1", 3: "b#b1"}
    assert len(trace_sources) == 3
    assert set(ChatService._extract_citation_ids(packed)) == set(mapping)


def test_unknown_compression_tag_cannot_claim_a_source():
    chunk = RetrievedChunk(doc_id="a", chunk_id="a1", content="A")
    context = ExtractiveCompressor.canonical_context(
        CompressionResult("[doc99-s1] Unknown"), [chunk],
        [{"doc_id": "a", "chunk_id": "a1", "citation_id": 1}],
    )
    assert "[doc" not in context
