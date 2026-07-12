"""
Terrain sampling engine for the UiTM Shah Alam DTM tiled imagery service.

The ArcGIS Online service stores numeric elevation tiles in LERC format.
This module:
- converts WGS84 coordinates to EPSG:32647
- locates the correct LERC tile
- decodes the tile
- returns elevation in metres
- samples a straight-line terrain profile between two points
"""

from __future__ import annotations

from functools import lru_cache
from math import floor
from typing import Iterable

import lerc
import numpy as np
import requests
from pyproj import Transformer


DTM_IMAGE_SERVICE_URL = (
    "https://tiledimageservices5.arcgis.com/2ZRAaoTSJbQ20ceg/"
    "arcgis/rest/services/DTM_UiTM_Shah_Alam_Analysis/ImageServer"
)

# Published service properties.
SERVICE_EPSG = 32647
TILE_SIZE = 256
TILE_ORIGIN_X = -5120763.26769827
TILE_ORIGIN_Y = 9997963.94301857

# The service has levels 0–6. Level 5 provides approximately 0.135 m pixels,
# which is already much finer than needed for a campus elevation profile.
DEFAULT_LEVEL = 5

LOD_RESOLUTIONS = {
    0: 4.31725439999996,
    1: 2.15862719999998,
    2: 1.07931359999999,
    3: 0.539656799999995,
    4: 0.269828399999998,
    5: 0.134914199999999,
    6: 0.0674570999999994,
}

WGS84_TO_DTM = Transformer.from_crs(
    "EPSG:4326",
    f"EPSG:{SERVICE_EPSG}",
    always_xy=True,
)


class TerrainSamplingError(RuntimeError):
    """Raised when an elevation tile cannot be sampled."""


@lru_cache(maxsize=512)
def _download_lerc_tile(
    level: int,
    tile_row: int,
    tile_column: int,
) -> bytes:
    """Download and cache one LERC tile."""

    url = (
        f"{DTM_IMAGE_SERVICE_URL}/tile/"
        f"{level}/{tile_row}/{tile_column}"
    )

    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        raise TerrainSamplingError(
            "ArcGIS elevation tile request failed: "
            f"HTTP {response.status_code}"
        )

    if not response.content:
        raise TerrainSamplingError(
            "ArcGIS returned an empty elevation tile."
        )

    return response.content


@lru_cache(maxsize=512)
def _decode_lerc_tile(
    level: int,
    tile_row: int,
    tile_column: int,
):
    """Decode one cached LERC tile into a NumPy masked array."""

    tile_bytes = _download_lerc_tile(
        level=level,
        tile_row=tile_row,
        tile_column=tile_column,
    )

    decoded = lerc.decode(tile_bytes)

    if not isinstance(decoded, tuple) or len(decoded) < 3:
        raise TerrainSamplingError(
            "The LERC decoder returned an unexpected result."
        )

    result_code, values, valid_mask = decoded[:3]

    if result_code != 0:
        raise TerrainSamplingError(
            f"LERC decoding failed with error code {result_code}."
        )

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[0]

    if values.ndim != 2:
        raise TerrainSamplingError(
            f"Unexpected elevation tile shape: {values.shape}"
        )

    if valid_mask is None:
        return np.ma.array(values, mask=False)

    valid_mask = np.asarray(valid_mask)

    if valid_mask.ndim == 3:
        valid_mask = valid_mask[0]

    return np.ma.array(
        values,
        mask=np.logical_not(valid_mask),
    )


def projected_to_tile_pixel(
    x: float,
    y: float,
    level: int = DEFAULT_LEVEL,
) -> tuple[int, int, int, int]:
    """Convert projected coordinates to tile row/column and pixel row/column."""

    if level not in LOD_RESOLUTIONS:
        raise ValueError(f"Unsupported level: {level}")

    resolution = LOD_RESOLUTIONS[level]
    tile_span = TILE_SIZE * resolution

    tile_column = floor(
        (x - TILE_ORIGIN_X) / tile_span
    )

    tile_row = floor(
        (TILE_ORIGIN_Y - y) / tile_span
    )

    tile_min_x = (
        TILE_ORIGIN_X
        + tile_column * tile_span
    )

    tile_max_y = (
        TILE_ORIGIN_Y
        - tile_row * tile_span
    )

    pixel_column = floor(
        (x - tile_min_x) / resolution
    )

    pixel_row = floor(
        (tile_max_y - y) / resolution
    )

    pixel_column = min(
        max(pixel_column, 0),
        TILE_SIZE - 1,
    )

    pixel_row = min(
        max(pixel_row, 0),
        TILE_SIZE - 1,
    )

    return (
        tile_row,
        tile_column,
        pixel_row,
        pixel_column,
    )


def sample_elevation_xy(
    x: float,
    y: float,
    level: int = DEFAULT_LEVEL,
) -> float:
    """Sample DTM elevation at one EPSG:32647 coordinate."""

    (
        tile_row,
        tile_column,
        pixel_row,
        pixel_column,
    ) = projected_to_tile_pixel(
        x=x,
        y=y,
        level=level,
    )

    tile = _decode_lerc_tile(
        level=level,
        tile_row=tile_row,
        tile_column=tile_column,
    )

    value = tile[
        pixel_row,
        pixel_column,
    ]

    if np.ma.is_masked(value):
        raise TerrainSamplingError(
            "The selected point is outside the valid DTM area."
        )

    elevation = float(value)

    if not np.isfinite(elevation):
        raise TerrainSamplingError(
            "The selected DTM cell has no valid elevation."
        )

    return elevation


def sample_elevation_wgs84(
    longitude: float,
    latitude: float,
    level: int = DEFAULT_LEVEL,
) -> float:
    """Sample DTM elevation at one longitude/latitude coordinate."""

    x, y = WGS84_TO_DTM.transform(
        longitude,
        latitude,
    )

    return sample_elevation_xy(
        x=x,
        y=y,
        level=level,
    )


def sample_profile_wgs84(
    start_longitude: float,
    start_latitude: float,
    end_longitude: float,
    end_latitude: float,
    sample_count: int = 120,
    level: int = DEFAULT_LEVEL,
) -> list[dict[str, float]]:
    """
    Sample a straight-line terrain profile between two WGS84 points.

    Distance is calculated in EPSG:32647 metres.
    """

    if sample_count < 2:
        raise ValueError(
            "sample_count must be at least 2."
        )

    start_x, start_y = WGS84_TO_DTM.transform(
        start_longitude,
        start_latitude,
    )

    end_x, end_y = WGS84_TO_DTM.transform(
        end_longitude,
        end_latitude,
    )

    x_values = np.linspace(
        start_x,
        end_x,
        sample_count,
    )

    y_values = np.linspace(
        start_y,
        end_y,
        sample_count,
    )

    total_distance = float(
        np.hypot(
            end_x - start_x,
            end_y - start_y,
        )
    )

    distances = np.linspace(
        0.0,
        total_distance,
        sample_count,
    )

    profile: list[dict[str, float]] = []

    for distance, x, y in zip(
        distances,
        x_values,
        y_values,
    ):
        try:
            elevation = sample_elevation_xy(
                x=float(x),
                y=float(y),
                level=level,
            )
        except TerrainSamplingError:
            elevation = float("nan")

        profile.append(
            {
                "Distance (m)": float(distance),
                "Elevation (m)": elevation,
            }
        )

    valid_count = sum(
        np.isfinite(item["Elevation (m)"])
        for item in profile
    )

    if valid_count < 2:
        raise TerrainSamplingError(
            "The terrain profile did not contain enough valid DTM values."
        )

    return profile
