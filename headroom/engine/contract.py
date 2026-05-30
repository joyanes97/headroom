from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class Provider(str, enum.Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


class Flavor(str, enum.Enum):
    MESSAGES = "messages"
    CHAT = "chat"
    RESPONSES = "responses"
    GENERATE = "generate"
    INVOKE = "invoke"
    RAW_PREDICT = "raw_predict"


@dataclass(frozen=True)
class RequestContext:
    provider: Provider
    flavor: Flavor
    headers_view: Mapping[str, str]
    raw_body: bytes
    session_key: str


@dataclass
class ResponseTelemetry:
    """Opaque to the host; routed to its own metrics sink. The serving front emits."""

    tokens_in: int = 0
    tokens_out: int = 0
    bytes_saved: int = 0
    compressed: bool = False
    ccr_fired: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestDecision:
    body: bytes
    telemetry: ResponseTelemetry
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamContext:
    session_key: str
    provider: Provider
    flavor: Flavor
    state: dict[str, Any] = field(default_factory=dict)
