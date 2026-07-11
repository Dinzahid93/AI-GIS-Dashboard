"""Gemini GIS command interpreter.

Install:
    pip install google-genai

The LLM only converts natural language into a structured GIS action.
All spatial calculations remain in the Streamlit GIS application.
"""

from __future__ import annotations

import json
from typing import Any


SUPPORTED_MODES = {
    "walking": "Walking",
    "walk": "Walking",
    "pedestrian": "Walking",
    "e-bike": "E-bike",
    "ebike": "E-bike",
    "e bike": "E-bike",
    "bike": "E-bike",
    "bicycle": "E-bike",
    "motorcycle": "Motorcycle",
    "motorbike": "Motorcycle",
    "car": "Car driving",
    "drive": "Car driving",
    "driving": "Car driving",
}


def _clean_result(result: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise Gemini's structured response."""

    action = str(result.get("action", "unknown")).strip()
    reply = str(result.get("reply", "")).strip()

    if action == "service_area":
        origin_id = int(result["origin_id"])
        minutes = float(result["minutes"])

        raw_modes = result.get("travel_modes") or ["Walking"]
        modes: list[str] = []
        for raw_mode in raw_modes:
            mode_text = str(raw_mode).strip()
            canonical = SUPPORTED_MODES.get(mode_text.lower(), mode_text)
            if canonical in {"Walking", "E-bike", "Motorcycle", "Car driving"}:
                if canonical not in modes:
                    modes.append(canonical)

        if not modes:
            modes = ["Walking"]

        if minutes <= 0:
            raise ValueError("Service-area minutes must be greater than zero.")

        return {
            "action": "service_area",
            "origin_id": origin_id,
            "minutes": minutes,
            "travel_modes": modes,
            "reply": reply or (
                f"Calculating a {minutes:g}-minute service area from "
                f"Building {origin_id} for {', '.join(modes)}."
            ),
        }

    if action == "multi_origin_route":
        origin_ids = [int(value) for value in result.get("origin_ids", [])]
        destination_id = int(result["destination_id"])
        return {
            "action": action,
            "origin_ids": origin_ids,
            "destination_id": destination_id,
            "reply": reply,
        }

    if action == "one_origin_multi_destination":
        origin_id = int(result["origin_id"])
        destination_ids = [
            int(value) for value in result.get("destination_ids", [])
        ]
        return {
            "action": action,
            "origin_id": origin_id,
            "destination_ids": destination_ids,
            "reply": reply,
        }

    if action == "route":
        building_ids = [int(value) for value in result.get("building_ids", [])]
        return {
            "action": "route",
            "building_ids": building_ids,
            "reply": reply,
        }

    return {
        "action": "unknown",
        "reply": reply or "I could not identify the requested GIS analysis.",
    }


def interpret_gis_command(
    question: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    """Use Gemini to classify a natural-language GIS request."""

    if not question or not question.strip():
        raise ValueError("The GIS question cannot be empty.")

    if not api_key or not api_key.strip():
        raise ValueError("A Gemini API key is required.")

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError(
            "The google-genai package is missing. Run: pip install google-genai"
        ) from error

    response_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "route",
                    "multi_origin_route",
                    "one_origin_multi_destination",
                    "service_area",
                    "unknown",
                ],
            },
            "building_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "origin_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "destination_id": {"type": ["integer", "null"]},
            "origin_id": {"type": ["integer", "null"]},
            "destination_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "minutes": {"type": ["number", "null"]},
            "travel_modes": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "Walking",
                        "E-bike",
                        "Motorcycle",
                        "Car driving",
                    ],
                },
            },
            "reply": {"type": "string"},
        },
        "required": ["action", "reply"],
    }

    system_instruction = """
You are a GIS command interpreter for a university campus network-analysis app.
Return only structured JSON that matches the supplied schema.

Choose exactly one action:

1. route
Use for one point to one point, or an ordered multi-stop journey.
Preserve the order in building_ids.
Example: "Start at Building 10, visit 20, then 35" -> [10, 20, 35].

2. multi_origin_route
Use when several origins each require an independent shortest path to one
common destination.
Example: "Separate routes from Buildings 1, 2 and 3 to Building 444".

3. one_origin_multi_destination
Use when one origin requires independent shortest paths to several destinations.
Example: "Separate routes from Building 10 to Buildings 20, 30 and 40".

4. service_area
Use for reachable-within-time or isochrone questions.
Extract origin_id, minutes and all requested travel_modes.
Examples:
- "Show all buildings within 5 minutes walking from Building 10"
- "Compare 5-minute walking, e-bike and car service areas from Building 10"
- "What can I reach in 8 minutes by motorcycle from Building 25?"
If no mode is stated for a service area, use Walking.

5. unknown
Use when the request does not contain enough information.

Do not calculate spatial results. Only classify the request and extract parameters.
""".strip()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_json_schema=response_schema,
            temperature=0,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise ValueError("Gemini did not return valid JSON.") from error

    return _clean_result(parsed)
