import random
import time
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ============================================================
# STRUCTURED GIS COMMAND
# ============================================================

class GISCommand(BaseModel):
    action: Literal["route", "unknown"] = Field(
        description=(
            "Use route when the user asks to travel, navigate, "
            "walk, find a path, or calculate a route between "
            "building FIDs. Otherwise use unknown."
        )
    )

    building_ids: list[int] = Field(
        description=(
            "Building FIDs in the exact requested visiting order."
        )
    )

    reply: str = Field(
        description=(
            "Briefly explain the interpreted request. "
            "Do not calculate distance, walking time, or coordinates."
        )
    )


SYSTEM_INSTRUCTION = """
You are the natural-language interface for a campus GIS.

The GIS currently supports shortest-path and multi-stop routing
between building FIDs.

Your only responsibilities are:
1. Detect whether the user wants a route.
2. Extract the building FIDs.
3. Preserve their requested visiting order.
4. Return structured JSON matching the supplied schema.

Rules:
- A building FID is an integer.
- At least two FIDs are needed for a route.
- Repeated FIDs are allowed for return journeys.
- Do not calculate route distance.
- Do not calculate walking time.
- Do not invent coordinates.
- The Python GIS engine performs the actual spatial analysis.

Examples:

User: Take me from Building 10 to Building 20
Result:
action = route
building_ids = [10, 20]

User: Start at 10, visit 20 and 35, then return to 10
Result:
action = route
building_ids = [10, 20, 35, 10]

User: What is the weather?
Result:
action = unknown
building_ids = []
"""


# Try the lightweight model first.
MODEL_LIST = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]


def _call_model(
    client,
    model_name: str,
    prompt: str,
) -> GISCommand:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GISCommand,
        ),
    )

    if not response.text:
        raise RuntimeError(
            f"{model_name} returned an empty response."
        )

    return GISCommand.model_validate_json(response.text)


def interpret_gis_command(
    question: str,
    api_key: str,
) -> dict:
    """
    Interpret natural language using Gemini.

    It tries multiple models and retries temporary server errors.
    The GIS engine still performs all route calculations.
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

    errors = []

    # Each model gets up to three attempts.
    for model_name in MODEL_LIST:
        for attempt in range(3):
            try:
                command = _call_model(
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                )

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

                result = command.model_dump()
                result["model_used"] = model_name

                return result

            except Exception as error:
                errors.append(
                    f"{model_name}, attempt {attempt + 1}: "
                    f"{error}"
                )

                # 1–2 sec, then 2–4 sec, then 4–8 sec.
                if attempt < 2:
                    delay = (2 ** attempt) + random.uniform(
                        0.5,
                        1.5,
                    )
                    time.sleep(delay)

    raise RuntimeError(
        "All Gemini models were temporarily unavailable. "
        + " | ".join(errors)
    )
