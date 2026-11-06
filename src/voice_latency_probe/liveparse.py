"""Parsing of what arrives over the data channel. Agents publish anything there."""

from __future__ import annotations

import json
from typing import Any


def parse_data_packet(raw: bytes) -> dict[str, Any] | None:
    """A JSON object, or None. A malformed packet must not end a measurement."""
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None
