import json
from pathlib import Path

import geopandas as gpd

from github_utils import (
    get_github_settings,
    update_file_on_github,
)


def ensure_name_column(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings = buildings.copy()

    if "NAME" not in buildings.columns:
        buildings["NAME"] = ""

    buildings["NAME"] = (
        buildings["NAME"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return buildings


def building_display_label(row) -> str:
    fid = int(row["FID"])
    name = str(row.get("NAME", "")).strip()

    if name:
        return f"{name} (FID {fid})"

    return f"Unnamed Building (FID {fid})"


def build_name_lookup(buildings: gpd.GeoDataFrame) -> dict:
    lookup = {}

    for _, row in buildings.iterrows():
        name = str(row.get("NAME", "")).strip()

        if name:
            lookup[name.casefold()] = int(row["FID"])

    return lookup


def update_building_name_locally(
    buildings: gpd.GeoDataFrame,
    building_fid: int,
    new_name: str,
) -> gpd.GeoDataFrame:
    updated = ensure_name_column(buildings)

    matching = updated["FID"] == int(building_fid)

    if not matching.any():
        raise ValueError(
            f"Building FID {building_fid} was not found."
        )

    cleaned_name = new_name.strip()

    if not cleaned_name:
        raise ValueError(
            "Building name cannot be empty."
        )

    duplicate = updated[
        updated["NAME"].str.casefold()
        == cleaned_name.casefold()
    ]

    duplicate = duplicate[
        duplicate["FID"] != int(building_fid)
    ]

    if not duplicate.empty:
        raise ValueError(
            f"The name '{cleaned_name}' is already used "
            f"by another building."
        )

    updated.loc[matching, "NAME"] = cleaned_name

    return updated


def save_building_names_to_github(
    buildings: gpd.GeoDataFrame,
    streamlit_secrets,
    building_fid: int,
    new_name: str,
) -> None:
    settings = get_github_settings(streamlit_secrets)

    geojson_text = buildings.to_json(
        drop_id=True,
        ensure_ascii=False,
    )

    geojson_data = json.loads(geojson_text)

    update_file_on_github(
        settings=settings,
        geojson_data=geojson_data,
        commit_message=(
            f"Update building {building_fid} name "
            f"to {new_name}"
        ),
    )
