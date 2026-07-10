from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class GISCommand(BaseModel):
    action: Literal["route", "unknown"] = Field(
        description=(
            "Use route only when the user requests navigation or "
            "a path between at least two building FIDs. "
            "Otherwise use unknown."
        )
    )
    building_ids: list[int] = Field(
        description=(
            "Building FIDs in the exact requested visiting order. "
            "Repeated IDs are allowed for return trips."
        )
    )
    reply: str = Field(
        description=(
            "A short explanation of what was understood. "
            "Do not calculate distance, walking time, coordinates, "
            "or the shortest path."
        )
    )


SYSTEM_INSTRUCTION = """
You are the natural-language command interpreter for a university campus GIS.

The current GIS supports only road-network routing between building FIDs.

Your job:
1. Determine whether the user requests a route.
2. Extract the building FIDs in the exact order requested.
3. Return only the structured response required by the schema.

Rules:
- A building FID is an integer.
- Preserve the user's requested visiting order.
- Multi-stop routes are allowed.
- Repeated FIDs are allowed for return trips.
- Do not calculate distance, walking time, coordinates, or routes.
- The Python GIS engine performs all spatial calculations.
- If fewer than two building FIDs are clearly provided, use action="unknown".
- Ignore unrelated numbers unless they clearly refer to building FIDs.

Examples:
User: Find the shortest route from Building 10 to Building 20
Result: action="route", building_ids=[10, 20]

User: Start at Building 10, visit 20 and 35, then return to 10
Result: action="route", building_ids=[10, 20, 35, 10]

User: What is the weather today?
Result: action="unknown", building_ids=[]
"""


def interpret_gis_command(question: str, api_key: str) -> dict:
    """Convert a natural-language request into a controlled GIS command."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a GIS request.")

    if not api_key:
        raise ValueError("Gemini API key is missing.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question.strip(),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0,
            response_mime_type="application/json",
            response_schema=GISCommand,
        ),
    )

    if getattr(response, "parsed", None) is not None:
        command = response.parsed
    else:
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")
        command = GISCommand.model_validate_json(response.text)

    building_ids = [int(value) for value in command.building_ids]

    if command.action == "route" and len(building_ids) < 2:
        return {
            "action": "unknown",
            "building_ids": [],
            "reply": "Please provide at least two building FIDs.",
        }

    if command.action == "unknown":
        building_ids = []

    return {
        "action": command.action,
        "building_ids": building_ids,
        "reply": command.reply.strip(),
    }
