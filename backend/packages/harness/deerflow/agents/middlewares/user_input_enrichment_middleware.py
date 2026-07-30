from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer

from deerflow.runtime.secret_context import USER_INPUT_ENRICHMENT_CONTEXT_KEY
from deerflow.utils.custom_events import emit_custom_event

logger = logging.getLogger(__name__)

# 与 input_sanitization_middleware 保持一致的分隔符
_USER_INPUT_BEGIN = "--- BEGIN USER INPUT ---"
_USER_INPUT_END = "--- END USER INPUT ---"


def _extract_raw_user_text(text: str) -> str:
    """从 prompt template 包装中提取原始用户输入。"""
    if _USER_INPUT_BEGIN in text:
        match = re.search(
            re.escape(_USER_INPUT_BEGIN) + r"\n?(.*?)\n?" + re.escape(_USER_INPUT_END),
            text,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
    return text.strip()


def _latest_user_message(messages: list) -> HumanMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not msg.additional_kwargs.get("hide_from_ui"):
            return msg
    return None


def _user_message_text(message: HumanMessage) -> str | None:
    content = message.content
    if isinstance(content, str) and content.strip():
        return _extract_raw_user_text(content)
    if isinstance(content, list):
        text_parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        joined = " ".join(text_parts).strip()
        if joined:
            return _extract_raw_user_text(joined)
    return None


def _user_message_key(message: HumanMessage, user_text: str) -> str:
    if message.id:
        return str(message.id)
    return hashlib.sha256(user_text.encode("utf-8")).hexdigest()


def _tool_result_text(result: Any) -> str:
    """Normalize LangChain tool output into a plain text/JSON string.

    MCP/LangChain tools often return content blocks, e.g.::

        [{'type': 'text', 'text': '{"code":200,"data":[...]}', 'id': '...'}]

    rather than a bare JSON string.
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        if isinstance(result.get("text"), str):
            return result["text"].strip()
        return json.dumps(result, ensure_ascii=False)
    if isinstance(result, list):
        parts: list[str] = []
        for block in result:
            if isinstance(block, str) and block.strip():
                parts.append(block.strip())
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                elif block:
                    parts.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(parts).strip()
    return str(result).strip()


def _parse_match_payload(raw: str) -> dict[str, Any] | None:
    """Parse match_for_best payload from tool text."""
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Tool text may wrap JSON with extra prose; extract the first object.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _format_match_knowledge(items: list[dict[str, Any]]) -> str:
    """Render matched entities for model context (not the raw transport envelope)."""
    lines: list[str] = []
    for item in items:
        name = item.get("name") or ""
        code = item.get("code") or ""
        entity_type = item.get("type") or ""
        similarity = item.get("similarity")
        entity_id = item.get("id")
        parts = [p for p in (entity_type, name, code) if p]
        label = " / ".join(str(p) for p in parts) if parts else str(entity_id or "unknown")
        if similarity is not None:
            lines.append(f"- {label} (similarity={similarity}, id={entity_id})")
        else:
            lines.append(f"- {label} (id={entity_id})")
    return "Matched entities:\n" + "\n".join(lines)


async def enrich(user_text: str) -> str:

    # 以mcp示例
    try:
        from deerflow.mcp.cache import initialize_mcp_tools

        tools = await initialize_mcp_tools()
    except Exception:
        logger.exception("[UserEnrichment] initialize_mcp_tools failed")
        return ""

    target = next((t for t in tools if t.name.endswith("match_for_best")), None)
    if not target:
        logger.warning(
            "[UserEnrichment] tool match_for_best not found in %d tools",
            len(tools),
        )
        return ""

    try:
        result = await target.ainvoke({"match_string": user_text})
        raw = _tool_result_text(result)
        logger.info("[UserEnrichment] match_for_best raw: %s", raw[:300])

        payload = _parse_match_payload(raw)
        if not payload:
            logger.warning("[UserEnrichment] unparseable match_for_best result")
            return ""

        if not payload.get("success", True):
            logger.info("[UserEnrichment] match_for_best success=false: %s", payload.get("message"))
            return ""

        data = payload.get("data") or []
        if not isinstance(data, list) or not data:
            return ""

        best = data[0] if isinstance(data[0], dict) else None
        if not best:
            return ""
        similarity = float(best.get("similarity") or 0.0)
        if similarity < 0.3:
            logger.info("[UserEnrichment] best similarity %.4f < 0.3, skip", similarity)
            return ""

        kept = [item for item in data if isinstance(item, dict) and float(item.get("similarity") or 0.0) >= 0.3]
        if not kept:
            return ""
        return _format_match_knowledge(kept)
    except Exception:
        logger.exception("[UserEnrichment] match_for_best call failed")
        return ""


class UserInputEnrichmentMiddleware(AgentMiddleware):
    """Inject matched entities into model requests without mutating graph state."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        message = _latest_user_message(list(request.messages))
        if message is None:
            return await handler(request)

        user_text = _user_message_text(message)
        if not user_text:
            return await handler(request)

        message_key = _user_message_key(message, user_text)
        context = request.runtime.context if isinstance(request.runtime.context, dict) else None
        cached = context.get(USER_INPUT_ENRICHMENT_CONTEXT_KEY) if context is not None else None
        if isinstance(cached, dict) and cached.get("message_key") == message_key:
            knowledge = cached.get("knowledge")
            knowledge = knowledge if isinstance(knowledge, str) else ""
        else:
            logger.info("[UserEnrichment] enriching model request: %s", user_text[:100])
            self._emit_status("started", user_text)
            failed = False
            try:
                knowledge = await enrich(user_text)
            except Exception:
                logger.exception("[UserEnrichment] enrichment failed")
                self._emit_status("failed", user_text)
                knowledge = ""
                failed = True

            knowledge = knowledge.strip()
            if context is not None:
                context[USER_INPUT_ENRICHMENT_CONTEXT_KEY] = {
                    "message_key": message_key,
                    "knowledge": knowledge,
                }
            if knowledge:
                logger.info("[UserEnrichment] injecting knowledge: %s", knowledge)
                self._emit_status("completed", user_text)
            elif not failed:
                logger.info("[UserEnrichment] empty knowledge, skipping injection")
                self._emit_status("skipped", user_text)

        if not knowledge:
            return await handler(request)

        extra = HumanMessage(
            content="<user_enrichment>\n" + knowledge + "\n</user_enrichment>",
            id=f"{message_key}__user_enrichment",
            additional_kwargs={"hide_from_ui": True},
        )
        return await handler(request.override(messages=[*request.messages, extra]))

    @staticmethod
    def _emit_status(status: str, user_text: str, **kwargs: Any) -> None:
        try:
            writer = get_stream_writer()
            payload = {
                "type": "user_enrichment",
                "status": status,
                "user_text": user_text,
                **kwargs,
            }
            emit_custom_event(payload, writer=writer)
            logger.info("[UserEnrichment] emitted event: status=%s", status)
        except Exception:
            logger.debug("[UserEnrichment] Failed to emit user_enrichment event", exc_info=True)
