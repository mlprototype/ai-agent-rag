from pathlib import Path
import sqlite3

import pytest

from domain.services.sqlite_structured_query import SQLiteDataSource
from infrastructure.sqlite.seed_structured_query_db import seed_db


@pytest.fixture(autouse=True)
def forbid_development_sqlite_access(monkeypatch):
    """通常のSQLite DBは、テストから読み取りも初期化もさせない。"""
    development_path = (Path(__file__).resolve().parents[1] / "data/structured_query.db").resolve()
    connect = sqlite3.connect

    def guarded_connect(database, *args, **kwargs):
        value = str(database)
        if value.startswith("file:"):
            value = value[5:].split("?", 1)[0]
        if value != ":memory:" and Path(value).resolve() == development_path:
            raise AssertionError("Tests must use an explicit temporary SQLite path")
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)


@pytest.fixture
def sqlite_datasource(tmp_path):
    db_path = tmp_path / "structured_query.db"
    seed_db(str(db_path))
    return SQLiteDataSource(str(db_path))
