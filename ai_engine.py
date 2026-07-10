from typing import Literal

from google import genai
from pydantic import BaseModel, Field


class GISCommand(BaseModel):
    action: Literal["route", "unknown"] = Field(
        description=(
            "Use 'route' when the user requests navigation, "
            "a path, or travel between building FIDs. "
            "Otherwise use 'unknown'."
        )
    )

    building_ids: list[int] = Field(
        description=(
            "Building FIDs in the exact order they should be visited. "
            "Return an empty list if the action is unknown."
        )
    )

    reply: str = Field(
        description=(
            "A short response explaining what was understood. "
            "Do not calculate distance or invent route results."
        )
    )


SYSTEM_INSTRUCTION = """
You are the natural-language interface for a university campus GIS.

The current GIS supports only road-network routing between building FIDs.

Your responsibility:
1. Interpret the user's request.
2. Extract building FIDs in the requested visiting order.
3. Return structured data only.

Rules:
- A building FID is an integer.
- Preserve the requested order.
- Multi-stop routes are allowed.
- Repeated IDs are allowed when returning to the starting building.
- Never calculate distance, walking time, coordinates, or the shortest path.
- The Python GIS engine performs all spatial calculations.
- If fewer than two building FIDs are provided, return action="unknown".
- Ignore unrelated numbers such as time, dates, and distances unless they
  clearly refer to building FIDs.

Examples:

User:
Find the shortest route from Building 10 to Building 20

Result:
action="route"
building_ids=[10, 20]

User:
Start at 10, visit 20 and 35, then return to 10

Result:
action="route"
building_ids=[10, 20, 35, 10]

User:
What is the weather today?

Result:
action="unknown"
building_ids=[]
"""


def interpret_gis_command(
    question: str,
    api_key: str,
) -> dict:
    """
    Use Gemini to convert a natural-language request into a
    controlled GIS command.

    Gemini only interprets intent and extracts parameters.
    The GIS engine still performs the spatial analysis.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a GIS request.")

    if not api_key:
        raise ValueError("Gemini API key is missing.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
{SYSTEM_INSTRUCTION}

User request:
{question.strip()}
"""

    interaction = client.interactions.create(
        model="gemini-3.5-flash",
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": GISCommand.model_json_schema(),
        },
    )

    command = GISCommand.model_validate_json(
        interaction.output_text
    )

    # Final safety validation
    command.building_ids = [
        int(building_id)
        for building_id in command.building_ids
    ]

    if (
        command.action == "route"
        and len(command.building_ids) < 2
    ):
        command.action = "unknown"
        command.building_ids = []
        command.reply = (
            "Please provide at least two building FIDs."
        )

    if command.action == "unknown":
        command.building_ids = []

    return command.model_dump()
