from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os
import tempfile

from domain.services.ingestion_service import IngestionService

router = APIRouter()

SUPPORTED_EXTENSIONS = {'.md', '.markdown', '.html', '.htm', '.txt'}

class IngestDirectoryRequest(BaseModel):
    """ディレクトリ内のファイルを一括取り込むためのリクエストモデル。"""
    directory_path: str = Field(description="取り込むファイルが含まれるディレクトリのパス")

class IngestResponse(BaseModel):
    """インジェスチョン結果のレスポンスモデル。"""
    message: str
    ingested_files: List[str] = []
    errors: List[str] = []

@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    アップロードされたファイルを取り込んでベクトルDBに登録します。
    対応フォーマット: Markdown (.md), HTML (.html), TXT (.txt)
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"サポートされていないファイル形式です: {ext}。対応形式: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    # アップロードされたファイルを一時ファイルに保存
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        IngestionService.ingest_file(tmp_path)
        return IngestResponse(
            message="ファイルの取り込みが完了しました。",
            ingested_files=[file.filename]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取り込み中にエラーが発生しました: {str(e)}")
    finally:
        # 一時ファイルを削除
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@router.post("/ingest/directory", response_model=IngestResponse)
async def ingest_directory(request: IngestDirectoryRequest):
    """
    指定されたディレクトリ内の対応ファイルを一括で取り込みます。
    """
    if not os.path.isdir(request.directory_path):
        raise HTTPException(status_code=400, detail=f"ディレクトリが見つかりません: {request.directory_path}")
    
    ingested = []
    errors = []
    
    for filename in os.listdir(request.directory_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        
        file_path = os.path.join(request.directory_path, filename)
        try:
            IngestionService.ingest_file(file_path)
            ingested.append(filename)
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
    
    return IngestResponse(
        message=f"{len(ingested)} 件のファイルを取り込みました。",
        ingested_files=ingested,
        errors=errors
    )
