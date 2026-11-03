"""Real MCP stdio smoke client. Start INKLAB first; this script never imports data."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]


def parameters():
    return StdioServerParameters(command=sys.executable, args=["-X", "utf8", str(ROOT / "mcp_server.py")], cwd=str(ROOT),
                                 env={"INKLAB_API_URL": os.environ.get("INKLAB_API_URL", "http://127.0.0.1:5003"),
                                      "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})


def unpack(result):
    if result.is_error:
        raise RuntimeError("MCP 工具调用失败：" + " ".join(part.text for part in result.content if hasattr(part, "text")))
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(next(part.text for part in result.content if hasattr(part, "text")))


async def smoke(query, record_id=None):
    async with Client(parameters(), read_timeout_seconds=60) as client:
        tools = await client.list_tools()
        resource = await client.read_resource("inklab://capabilities")
        documents = unpack(await client.call_tool("list_knowledge_documents", {}))
        search = unpack(await client.call_tool("search_knowledge", {"query": query, "limit": 3}))
        answer = unpack(await client.call_tool("ask_knowledge", {"question": query}))
        prompt = await client.get_prompt("study_query", {"question": query})
        result = {"success": True, "protocol_version": client.protocol_version, "transport": "real stdio → local HTTP",
                  "tools": [item.name for item in tools.tools], "resource_count": len(resource.contents),
                  "prompt_messages": len(prompt.messages), "documents": len(documents["documents"]),
                  "sources": search["sources"], "answer_mode": answer["mode"], "llm_used": answer["llm_used"],
                  "reply": answer["reply"]}
        if record_id is not None:
            result["ocr_record"] = unpack(await client.call_tool("get_ocr_record", {"record_id": record_id}))["record"]
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="为什么要先核对识别文字再导入知识库？")
    parser.add_argument("--record-id", type=int)
    args = parser.parse_args()
    if args.record_id is not None and args.record_id < 1:
        parser.error("--record-id must be positive")
    try:
        report = asyncio.run(smoke(args.query, args.record_id))
    except Exception as error:
        # Expected MCP error messages are bounded and scrubbed by the server.
        print(json.dumps({"success": False, "error": str(error)[:800]}, ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps(report, ensure_ascii=False, indent=2))
