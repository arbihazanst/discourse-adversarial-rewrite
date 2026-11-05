"""
title: Decision Models - Clef and d3
version: 0.1.0
requirements: httpx

Open WebUI Pipe function: exposes Clef and d3 as separate selectable models.
Both use llama.cpp's /v1/systemone API for binary spam classification.
Calls originate on the Open WebUI server, NOT the user's local PC.
GPU-server HOST port 30005 -> decision-clef container port 8080.
GPU-server HOST port 30006 -> decision-d3 container port 8080.
The Pipe uses Docker-internal container names and ports; this is intentional.
Ensure Open WebUI and both containers share a Docker network.
"""

import json
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        CLEF_BASE_URL: str = Field(
            default="http://decision-clef:8080",
            description="Clef URL reachable from the Open WebUI backend",
        )
        D3_BASE_URL: str = Field(
            default="http://decision-d3:8080",
            description="d3 URL reachable from the Open WebUI backend",
        )
        TIMEOUT_SECONDS: int = Field(default=180, ge=10, le=600)

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [
            {"id": "clef", "name": "Clef 27B - Spam Classifier"},
            {"id": "d3", "name": "d3 27B - Spam Classifier"},
        ]

    @staticmethod
    def _last_user_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                return "\n".join(parts).strip()
        return ""

    async def pipe(self, body: dict):
        if body.get("stream", False):
            raise ValueError("Use stream=false: System One classification is non-streaming.")

        selected = str(body.get("model", "")).split(".", 1)[-1]
        endpoints = {
            "clef": self.valves.CLEF_BASE_URL,
            "d3": self.valves.D3_BASE_URL,
        }
        if selected not in endpoints:
            raise ValueError(f"Unsupported decision model: {selected}")

        email_text = self._last_user_text(body.get("messages", []))
        if not email_text:
            raise ValueError("Provide the email to classify in the user message.")

        request = {
            "state": email_text,
            "questions": {
                "spam": {
                    "type": "choice",
                    "instructions": "Classify this email as SPAM or NOT_SPAM based on its content.",
                    "criteria": {
                        "SPAM": "Unsolicited advertising, phishing, scams, fraudulent offers, or deceptive messages.",
                        "NOT_SPAM": "Legitimate personal, business, or informational correspondence.",
                    },
                }
            },
        }
        url = endpoints[selected].rstrip("/") + "/v1/systemone"
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.valves.TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=request)
            response.raise_for_status()
            result = response.json()

        answer = result.get("answers", {}).get("spam", {})
        label = answer.get("choice")
        if label not in ("SPAM", "NOT_SPAM"):
            raise RuntimeError(f"Invalid System One classification response: {result}")

        output = {
            "model": selected,
            "label": label,
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence"),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "usage": result.get("usage"),
        }
        return json.dumps(output, ensure_ascii=False)
