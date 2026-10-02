import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["evaluation.evaluate", "scripts.check_confidence"])
def test_import_without_api_keys_does_not_initialize_llm_or_run_manual_check(module, tmp_path):
    environment = dict(os.environ)
    for name in ("OPENAI_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "COHERE_API_KEY"):
        environment.pop(name, None)
    environment.update({
        "PYTHON_DOTENV_DISABLED": "1",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_TRACING": "false",
        "ENABLE_RERANK": "false",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    })
    probe = """
import asyncio
import importlib
import socket
import sys
from unittest.mock import patch

with (
    patch('langchain_openai.ChatOpenAI', side_effect=AssertionError('LLM initialized on import')),
    patch('asyncio.run', side_effect=AssertionError('Manual check executed on import')),
    patch.object(socket.socket, 'connect', side_effect=AssertionError('Network used on import')),
):
    importlib.import_module(sys.argv[1])
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, module], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
