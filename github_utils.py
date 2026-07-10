import base64
import json
from typing import Any

import requests


def get_github_settings(secrets: Any) -> dict:
    required_keys = [
        "GITHUB_TOKEN",
        "GITHUB_OWNER",
        "GITHUB_REPO",
        "GITHUB_BRANCH",
        "GITHUB_FILE_PATH",
    ]

    missing = [
        key
        for key in required_keys
        if key not in secrets or not str(secrets[key]).strip()
    ]

    if missing:
        raise ValueError(
            f"Missing Streamlit secrets: {', '.join(missing)}"
        )

    return {
        "token": str(secrets["GITHUB_TOKEN"]),
        "owner": str(secrets["GITHUB_OWNER"]),
        "repo": str(secrets["GITHUB_REPO"]),
        "branch": str(secrets["GITHUB_BRANCH"]),
        "file_path": str(secrets["GITHUB_FILE_PATH"]),
    }


def get_file_from_github(settings: dict) -> dict:
    url = (
        f"https://api.github.com/repos/"
        f"{settings['owner']}/{settings['repo']}/contents/"
        f"{settings['file_path']}"
    )

    headers = {
        "Authorization": f"Bearer {settings['token']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(
        url,
        headers=headers,
        params={"ref": settings["branch"]},
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Unable to read GitHub file: "
            f"{response.status_code} {response.text}"
        )

    return response.json()


def update_file_on_github(
    settings: dict,
    geojson_data: dict,
    commit_message: str,
) -> None:
    current_file = get_file_from_github(settings)

    encoded_content = base64.b64encode(
        json.dumps(
            geojson_data,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
    ).decode("utf-8")

    url = (
        f"https://api.github.com/repos/"
        f"{settings['owner']}/{settings['repo']}/contents/"
        f"{settings['file_path']}"
    )

    headers = {
        "Authorization": f"Bearer {settings['token']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    payload = {
        "message": commit_message,
        "content": encoded_content,
        "sha": current_file["sha"],
        "branch": settings["branch"],
    }

    response = requests.put(
        url,
        headers=headers,
        json=payload,
        timeout=30,
    )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Unable to update GitHub file: "
            f"{response.status_code} {response.text}"
        )
