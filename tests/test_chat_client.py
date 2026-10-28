"""本地 Chat Completions 传输和协议解析。"""

import io
import json
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from recagent.agent.chat_client import (
    ModelUnavailableError,
    OpenAICompatibleClient,
    ProtocolError,
)


def test_client_sends_tools_and_parses_tool_call():
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "recall_candidates",
                                "arguments": '{"genres":["Sci-Fi"]}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }
    client = OpenAICompatibleClient(
        "http://127.0.0.1:11434/v1", "qwen2.5:1.5b", api_key="local-secret"
    )
    with patch(
        "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())
    ) as mock:
        turn = client.complete([{"role": "user", "content": "科幻"}], [{"type": "function"}], 7.5)
    request = mock.call_args.args[0]
    body = json.loads(request.data)
    assert request.full_url == "http://127.0.0.1:11434/v1/chat/completions"
    assert body["model"] == "qwen2.5:1.5b"
    assert body["parallel_tool_calls"] is False
    assert body["tools"] == [{"type": "function"}]
    assert mock.call_args.kwargs["timeout"] == 7.5
    assert turn.tool_calls[0].name == "recall_candidates"
    assert turn.tool_calls[0].arguments == '{"genres":["Sci-Fi"]}'
    assert turn.usage["total_tokens"] == 20


@pytest.mark.parametrize("bad_url", ["https://example.com/v1", "http://192.168.1.2:8000/v1"])
def test_client_rejects_nonlocal_endpoint(bad_url):
    with pytest.raises(ValueError):
        OpenAICompatibleClient(bad_url, "model")


def test_client_reports_unavailable_without_leaking_key():
    client = OpenAICompatibleClient("http://localhost:11434/v1", "model", api_key="secret-value")
    with (
        patch("urllib.request.urlopen", side_effect=URLError("connection refused")),
        pytest.raises(ModelUnavailableError) as error,
    ):
        client.complete([], [], 2)
    assert "secret-value" not in str(error.value)


def test_client_rejects_invalid_response():
    client = OpenAICompatibleClient("http://localhost:11434/v1", "model")
    with (
        patch("urllib.request.urlopen", return_value=io.BytesIO(b'{"choices":[]}')),
        pytest.raises(ProtocolError),
    ):
        client.complete([], [], 2)
    http_error = HTTPError("http://localhost:11434/v1", 500, "secret-value", {}, None)
    with (
        patch("urllib.request.urlopen", side_effect=http_error),
        pytest.raises(ModelUnavailableError) as error,
    ):
        client.complete([], [], 2)
    assert "secret-value" not in str(error.value)
