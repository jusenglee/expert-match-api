"""
Helpers for extracting the first JSON object from LLM responses.
"""

from __future__ import annotations

from typing import Any


def _strip_xml_block(text: str, tag_name: str) -> str:
    lower_text = text.lower()
    open_tag = f"<{tag_name.lower()}>"
    close_tag = f"</{tag_name.lower()}>"
    parts: list[str] = []
    cursor = 0

    while True:
        start = lower_text.find(open_tag, cursor)
        if start == -1:
            parts.append(text[cursor:])
            break
        parts.append(text[cursor:start])
        end = lower_text.find(close_tag, start + len(open_tag))
        if end == -1:
            break
        cursor = end + len(close_tag)

    return "".join(parts).strip()


def _extract_code_fence_body(text: str) -> str:
    fence = "```"
    start = text.find(fence)
    if start == -1:
        return text

    end = text.find(fence, start + len(fence))
    if end == -1:
        return text.replace(fence, "").strip()

    body_start = start + len(fence)
    if text[body_start : body_start + 4].lower() == "json":
        body_start += 4
    body = text[body_start:end].lstrip("\r\n").rstrip()
    return body or text.replace(fence, "").strip()


def extract_json_object_text(content: Any) -> str:
    """Extract the first JSON object from a model response."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        text = "".join(parts)
    else:
        text = str(content)

    text = _strip_xml_block(text.strip(), "thinking")
    text = _extract_code_fence_body(text)

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape_next = False
    for index in range(start, len(text)):
        char = text[index]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            if in_string:
                escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text
