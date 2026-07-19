"""1件の質問を実行し、spec-rag-qa互換AgentRunTrace JSONを保存するCLI。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from application.dto.chat_models import ChatRequest
from application.services.chat_service import ChatService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True, help="評価ケースID")
    parser.add_argument("--question", required=True, help="Agentへ渡す質問")
    parser.add_argument("--output", required=True, type=Path, help="Trace JSONの出力先")
    parser.add_argument("--session-id", help="会話セッションID（省略時は一意なIDを生成）")
    parser.add_argument("--target", default="ai-agent-rag", help="評価対象名")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> Path:
    session_id = args.session_id or f"agent-eval-{uuid4()}"
    request = ChatRequest(session_id=session_id, question=args.question)
    trace = await ChatService.create_agent_run_trace(
        request,
        case_id=args.case_id,
        target=args.target,
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        trace.model_dump_json(indent=2, exclude_unset=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    output_path = asyncio.run(run(parse_args()))
    print(output_path)


if __name__ == "__main__":
    main()
