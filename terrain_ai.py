"""
LLM interpreter for terrain GIS commands.

Gemini converts natural-language requests into a structured action.
All elevation calculations remain in the local GIS terrain engine.
"""

from __future__ import annotations

import json
import re
from typing import Any


SUPPORTED_ACTIONS = {
    "building_elevation",
    "compare_building_elevations",
    "elevation_profile",
    "unknown",
}


def _extract_json(text: str) -> dict[str, Any]:
    """Extract one JSON object from a model response."""

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "The model response did not contain JSON."
        )

    return json.loads(
        cleaned[start : end + 1]
    )


def fallback_terrain_parser(
    question: str,
) -> dict[str, Any]:
    """Deterministic fallback when Gemini is unavailable."""

    text = question.lower().strip()

    building_ids = [
        int(value)
        for value in re.findall(
            r"building\s*(\d+)",
            text,
        )
    ]

    if (
        any(
            phrase in text
            for phrase in [
                "profile",
                "elevation profile",
                "terrain profile",
            ]
        )
        and len(building_ids) >= 2
    ):
        return {
            "action": "elevation_profile",
            "origin_id": building_ids[0],
            "destination_id": building_ids[1],
            "reply": (
                "Generating a DTM terrain profile between "
                f"Building {building_ids[0]} and "
                f"Building {building_ids[1]}."
            ),
        }

    if (
        any(
            phrase in text
            for phrase in [
                "compare",
                "difference",
                "higher",
                "lower",
            ]
        )
        and len(building_ids) >= 2
    ):
        return {
            "action": "compare_building_elevations",
            "building_ids": building_ids[:2],
            "reply": (
                "Comparing DTM elevation at "
                f"Buildings {building_ids[0]} and "
                f"{building_ids[1]}."
            ),
        }

    if (
        any(
            phrase in text
            for phrase in [
                "elevation",
                "height",
                "terrain level",
                "ground level",
            ]
        )
        and len(building_ids) >= 1
    ):
        return {
            "action": "building_elevation",
            "building_id": building_ids[0],
            "reply": (
                "Retrieving DTM ground elevation for "
                f"Building {building_ids[0]}."
            ),
        }

    return {
        "action": "unknown",
        "reply": (
            "Try asking for a building elevation, an elevation comparison, "
            "or an elevation profile between two buildings."
        ),
    }


def interpret_terrain_command(
    question: str,
    api_key: str,
    model_name: str,
) -> dict[str, Any]:
    """Use Gemini to interpret a terrain GIS request."""

    if not question.strip():
        return {
            "action": "unknown",
            "reply": "Please enter a terrain GIS request.",
        }

    if not api_key.strip():
        return fallback_terrain_parser(question)

    try:
        from google import genai

        client = genai.Client(
            api_key=api_key,
        )

        prompt = f"""
You are a GIS command interpreter.

Convert the user's terrain request into JSON only.

Supported actions:

1. building_elevation
{{
  "action": "building_elevation",
  "building_id": 10,
  "reply": "Retrieving DTM ground elevation for Building 10."
}}

2. compare_building_elevations
{{
  "action": "compare_building_elevations",
  "building_ids": [10, 20],
  "reply": "Comparing DTM elevation at Buildings 10 and 20."
}}

3. elevation_profile
{{
  "action": "elevation_profile",
  "origin_id": 10,
  "destination_id": 20,
  "reply": "Generating a DTM terrain profile between Buildings 10 and 20."
}}

4. unknown
{{
  "action": "unknown",
  "reply": "The terrain request is not supported."
}}

Rules:
- Extract building numbers exactly.
- Use only the supported actions.
- Do not calculate or invent elevation values.
- Return valid JSON only.

User request:
{question}
"""

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
            },
        )

        parsed = _extract_json(
            str(response.text)
        )

        action = parsed.get(
            "action",
            "unknown",
        )

        if action not in SUPPORTED_ACTIONS:
            raise ValueError(
                f"Unsupported action returned: {action}"
            )

        return parsed

    except Exception:
        return fallback_terrain_parser(question)
