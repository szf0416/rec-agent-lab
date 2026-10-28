"""对本机 OpenAI 兼容 Chat Completions 服务的最小传输层。"""

from __future__ import annotations

import ipaddress
import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit


class ModelUnavailableError(RuntimeError):
    """本机模型服务没有返回可用响应。"""


class ProtocolError(RuntimeError):
    """模型响应不符合 Chat Completions 协议。"""


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ChatTurn:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    usage: dict[str, int]


class ChatClient(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout_s: float
    ) -> ChatTurn: ...


class OpenAICompatibleClient:
    def __init__(
        self, base_url: str, model: str, api_key: str | None = None, max_tokens: int = 512
    ) -> None:
        parsed = urlsplit(base_url)
        host = parsed.hostname
        try:
            loopback = host == "localhost" or (
                host is not None and ipaddress.ip_address(host).is_loopback
            )
        except ValueError:
            loopback = False
        if (
            parsed.scheme != "http"
            or not loopback
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"
        ):
            raise ValueError("模型地址必须是本机 HTTP /v1 端点")
        if not model.strip() or not 1 <= max_tokens <= 4096:
            raise ValueError("model 或 max_tokens 无效")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout_s: float
    ) -> ChatTurn:
        if timeout_s <= 0:
            raise ValueError("timeout_s 必须为正数")
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "parallel_tool_calls": False,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError):
            raise ModelUnavailableError("本机模型服务不可用或请求失败") from None
        except (ValueError, UnicodeError):
            raise ProtocolError("模型响应不是有效 JSON") from None
        return self._parse(payload)

    @staticmethod
    def _parse(payload: Any) -> ChatTurn:
        try:
            message = payload["choices"][0]["message"]
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise TypeError
            raw_calls = message.get("tool_calls") or []
            if not isinstance(raw_calls, list):
                raise TypeError
            calls = []
            for raw in raw_calls:
                if raw["type"] != "function":
                    raise TypeError
                call_id = raw["id"]
                name = raw["function"]["name"]
                arguments = raw["function"]["arguments"]
                if not all(isinstance(value, str) and value for value in (call_id, name)):
                    raise TypeError
                if not isinstance(arguments, str):
                    raise TypeError
                calls.append(ToolCall(call_id, name, arguments))
            usage_raw = payload.get("usage") or {}
            if not isinstance(usage_raw, dict):
                raise TypeError
            usage = {
                key: value
                for key, value in usage_raw.items()
                if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                and type(value) is int
                and value >= 0
            }
            return ChatTurn(content, tuple(calls), usage)
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProtocolError("模型响应字段无效") from None
