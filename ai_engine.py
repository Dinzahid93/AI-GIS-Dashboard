from typing import Literal

from google import genai
from pydantic import BaseModel, Field


# ============================================================
# GEMINI STRUCTURED RESPONSE
# ============================================================

class GISCommand(BaseModel):
    action: Literal["route", "unknown"] = Field(
        description=(
            "Use 'route' when the user asks for navigation, "
            "shortest path, travel, walking, or routing between "
            "building FIDs. Otherwise return 'unknown'."
        )
    )

    building_ids: list[int] = Field(
        description=(
            "Building FIDs in the exact order requested by the user. "
            "Return an empty list when the action is unknown."
        )
    )

    reply: str = Field(
        description=(
            "A brief explanation of what was understood. "
            "Do not calculate distance, time, coordinates, or routes."
        )
    )


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are the natural-language interface for a university campus GIS.

The GIS currently supports road-network routing between building FIDs.

Your task is only to:
1. Understand the user's request.
2. Identify whether the user wants a route.
3. Extract building FIDs in the exact visiting order.
4. Return structured JSON matching the required schema.

Important rules:
- Building FIDs are integers.
- Preserve the order requested by the user.
- Multi-stop routes are supported.
- Repeated FIDs are allowed when returning to the starting building.
- Never calculate route distance.
- Never calculate walking time.
- Never invent coordinates.
- Never calculate the shortest path yourself.
- The Python GIS engine performs all spatial calculations.
- If fewer than two building FIDs are clearly provided, return:
  action="unknown"
  building_ids=[]
- Ignore numbers representing times, dates, distances, or quantities
  unless they clearly refer to building FIDs.

Examples:

User:
Find the shortest route from Building 10 to Building 20

Result:
action="route"
building_ids=[10, 20]

User:
Take me from building 10 to building 20 and then building 35

Result:
action="route"
building_ids=[10, 20, 35]

User:
Start at Building 10, visit Building 20, then return to Building 10

Result:
action="route"
building_ids=[10, 20, 10]

User:
I need to visit 15, then 30, then 40

Result:
action="route"
building_ids=[15, 30, 40]

User:
What is the weather today?

Result:
action="unknown"
building_ids=[]
"""


# ============================================================
# GEMINI INTERPRETER
# ============================================================

def interpret_gis_command(
    question: str,
    api_key: str,
) -> dict:
    """
    Convert a natural-language request into a controlled GIS command.

    Gemini only determines the GIS action and extracts building FIDs.
    NetworkX and GeoPandas perform the actual spatial analysis.
    """

    if not isinstance(question, str):
        raise TypeError("The GIS request must be text.")

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Please enter a GIS request.")

    if not api_key or not str(api_key).strip():
        raise ValueError("Gemini API key is missing.")

    client = genai.Client(
        api_key=str(api_key).strip()
    )

    prompt = f"""
{SYSTEM_INSTRUCTION}

User request:
{cleaned_question}
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

    if not interaction.output_text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    command = GISCommand.model_validate_json(
        interaction.output_text
    )

    # Convert all returned IDs safely into integers
    command.building_ids = [
        int(building_id)
        for building_id in command.building_ids
    ]

    # Final safety check
    if (
        command.action == "route"
        and len(command.building_ids) < 2
    ):
        command.action = "unknown"
        command.building_ids = []
        command.reply = (
            "Please provide at least two building FIDs "
            "for route analysis."
        )

    if command.action == "unknown":
        command.building_ids = []

    return command.model_dump()
