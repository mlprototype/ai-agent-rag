from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from api.routers import ingest


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["read", "creation", "write", "ingestion", None])
async def test_upload_cleanup_keeps_primary_error_and_removes_created_file(tmp_path, failure):
    path = tmp_path / "upload.txt"
    temporary = Mock()
    temporary.name = str(path)
    temporary.write.side_effect = OSError("primary write") if failure == "write" else None
    context = Mock()
    context.__enter__ = Mock(return_value=temporary)
    context.__exit__ = Mock(return_value=False)
    if failure != "creation":
        path.touch()
    file = SimpleNamespace(filename="upload.txt", read=AsyncMock(return_value=b"text", side_effect=OSError("primary read") if failure == "read" else None))
    with patch.object(ingest.tempfile, "NamedTemporaryFile", return_value=context, side_effect=OSError("primary creation") if failure == "creation" else None), patch.object(ingest.IngestionService, "ingest_file", side_effect=OSError("primary ingestion") if failure == "ingestion" else None):
        if failure:
            with pytest.raises(HTTPException) as caught:
                await ingest.ingest_file(file)
            assert caught.value.status_code == 500
            assert f"primary {failure}" in caught.value.detail
        else:
            result = await ingest.ingest_file(file)
            assert result.ingested_files == ["upload.txt"]
    assert not path.exists()


@pytest.mark.anyio
async def test_cleanup_failure_does_not_mask_read_error(tmp_path, caplog):
    temporary = Mock(name=str(tmp_path / "upload.txt"))
    temporary.name = str(tmp_path / "upload.txt")
    context = Mock()
    context.__enter__ = Mock(return_value=temporary)
    context.__exit__ = Mock(return_value=False)
    file = SimpleNamespace(filename="upload.txt", read=AsyncMock(side_effect=OSError("primary read")))
    with patch.object(ingest.tempfile, "NamedTemporaryFile", return_value=context), patch.object(ingest.os, "unlink", side_effect=OSError("cleanup")):
        with pytest.raises(HTTPException) as caught:
            await ingest.ingest_file(file)
    assert "primary read" in caught.value.detail
    assert "cleanup failed" in caplog.text
