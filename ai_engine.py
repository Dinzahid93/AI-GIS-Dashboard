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
        origin_value = result.get("origin_id")
        minutes_value = result.get("minutes")

        if origin_value is None:
            raise ValueError(
                "Gemini did not provide an origin building for the service area."
            )

        if minutes_value is None:
            raise ValueError(
                "Gemini did not provide a travel-time limit for the service area."
            )

        origin_id = int(origin_value)
        minutes = float(minutes_value)

        raw_modes = result.get("travel_modes") or ["Walking"]
        modes: list[str] = []

        for raw_mode in raw_modes:
            mode_text = str(raw_mode).strip()
            canonical = SUPPORTED_MODES.get(
                mode_text.lower(),
                mode_text,
            )

            if canonical in {
                "Walking",
                "E-bike",
                "Motorcycle",
                "Car driving",
            }:
                if canonical not in modes:
                    modes.append(canonical)

        if not modes:
            modes = ["Walking"]

        if minutes <= 0:
            raise ValueError(
                "Service-area minutes must be greater than zero."
            )

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
        origin_ids = [
            int(value)
            for value in result.get("origin_ids", [])
        ]

        destination_value = result.get("destination_id")

        if not origin_ids:
            raise ValueError(
                "Gemini did not provide origin buildings."
            )

        if destination_value is None:
            raise ValueError(
                "Gemini did not provide a destination building."
            )

        destination_id = int(destination_value)

        return {
            "action": "multi_origin_route",
            "origin_ids": origin_ids,
            "destination_id": destination_id,
            "reply": reply or (
                f"Calculating separate shortest paths from Buildings "
                f"{origin_ids} to Building {destination_id}."
            ),
        }

    if action == "one_origin_multi_destination":
        origin_value = result.get("origin_id")
        destination_ids = [
            int(value)
            for value in result.get("destination_ids", [])
        ]

        if origin_value is None:
            raise ValueError(
                "Gemini did not provide an origin building."
            )

        if not destination_ids:
            raise ValueError(
                "Gemini did not provide destination buildings."
            )

        origin_id = int(origin_value)

        return {
            "action": "one_origin_multi_destination",
            "origin_id": origin_id,
            "destination_ids": destination_ids,
            "reply": reply or (
                f"Calculating separate shortest paths from Building "
                f"{origin_id} to Buildings {destination_ids}."
            ),
        }

    if action == "route":
        building_ids = [
            int(value)
            for value in result.get("building_ids", [])
        ]

        if len(building_ids) < 2:
            raise ValueError(
                "Gemini did not provide at least two buildings for routing."
            )

        return {
            "action": "route",
            "building_ids": building_ids,
            "reply": reply or (
                f"Calculating the route through Buildings "
                f"{building_ids} in the requested order."
            ),
        }

    return {
        "action": "unknown",
        "reply": reply or (
            "I could not identify the requested GIS analysis."
        ),
    }


def interpret_gis_command(
    question: str,
    api_key: str,
    model_name: str = "gemini-3.1-flash-lite",
) -> dict[str, Any]:
    """
    Use Gemini to classify a natural-language GIS request.

    Supported actions:
    - route
    - multi_origin_route
    - one_origin_multi_destination
    - service_area
    - unknown
    """

    if not question or not question.strip():
        raise ValueError(
            "The GIS question cannot be empty."
        )

    if not api_key or not api_key.strip():
        raise ValueError(
            "A Gemini API key is required."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError(
            "The google-genai package is missing. "
            "Run: pip install google-genai"
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
                "items": {
                    "type": "integer",
                },
            },
            "origin_ids": {
                "type": "array",
                "items": {
                    "type": "integer",
                },
            },
            "destination_id": {
                "type": [
                    "integer",
                    "null",
                ],
            },
            "origin_id": {
                "type": [
                    "integer",
                    "null",
                ],
            },
            "destination_ids": {
                "type": "array",
                "items": {
                    "type": "integer",
                },
            },
            "minutes": {
                "type": [
                    "number",
                    "null",
                ],
            },
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
            "reply": {
                "type": "string",
            },
        },
        "required": [
            "action",
            "reply",
        ],
    }

    system_instruction = """
You are a GIS command interpreter for a university campus
network-analysis application.

Return only structured JSON matching the supplied schema.

Choose exactly one action.

1. route

Use for:
- one building to one building; or
- an ordered multi-stop journey.

Preserve the exact requested order in building_ids.

Examples:
"Find the shortest route from Building 10 to Building 20."
building_ids = [10, 20]

"Start at Building 10, visit Buildings 20 and 35,
then go to Building 50."
building_ids = [10, 20, 35, 50]


2. multi_origin_route

Use when several origin buildings each require an independent
shortest path to one common destination.

Example:
"Show separate routes from Buildings 1, 2 and 3
to Building 444."

origin_ids = [1, 2, 3]
destination_id = 444


3. one_origin_multi_destination

Use when one origin building requires independent shortest paths
to several destination buildings.

Example:
"Show separate routes from Building 10
to Buildings 20, 30 and 40."

origin_id = 10
destination_ids = [20, 30, 40]


4. service_area

Use for:
- reachable-within-time requests;
- service-area requests;
- isochrone requests; or
- comparisons of accessibility by travel mode.

Extract:
- origin_id
- minutes
- every requested travel mode

Allowed travel modes:
- Walking
- E-bike
- Motorcycle
- Car driving

Examples:
"Show all buildings within 5 minutes walking
from Building 10."

origin_id = 10
minutes = 5
travel_modes = ["Walking"]

"Compare 5-minute walking, e-bike and car-driving
service areas from Building 10."

origin_id = 10
minutes = 5
travel_modes = [
    "Walking",
    "E-bike",
    "Car driving"
]

"What buildings can I reach within 8 minutes
by motorcycle from Building 25?"

origin_id = 25
minutes = 8
travel_modes = ["Motorcycle"]

If the user requests a service area without naming a travel mode,
use Walking.


5. unknown

Use only when the request does not contain enough information
to perform one of the supported GIS analyses.

Do not perform spatial calculations.
Do not invent building IDs, time limits, or travel modes.
Only classify the request and extract its parameters.
""".strip()

    client = genai.Client(
        api_key=api_key.strip(),
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=question.strip(),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_json_schema=response_schema,
                temperature=0,
            ),
        )
    except Exception as error:
        raise RuntimeError(
            f"Gemini request failed using model "
            f"'{model_name}': {error}"
        ) from error

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    try:
        parsed = json.loads(
            response.text
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "Gemini did not return valid JSON."
        ) from error

    if not isinstance(parsed, dict):
        raise ValueError(
            "Gemini returned JSON in an unexpected format."
        )

    return _clean_result(
        parsed
    )
