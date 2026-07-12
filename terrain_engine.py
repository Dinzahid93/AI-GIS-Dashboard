"""
Terrain raster sampling engine for the UiTM Shah Alam dashboard.

This version reads the lightweight GeoTIFF stored in the GitHub repository:

    data/here.tif

It supports:
- DTM elevation at one longitude/latitude location
- DTM elevation at one projected coordinate
- Straight-line elevation profiles between two locations
- Automatic coordinate transformation from WGS84 to the raster CRS

The raster remains on disk and is sampled only where needed. The full
GeoTIFF is not loaded into memory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.io import DatasetReader


# ============================================================
# 1. FILE LOCATION
# ============================================================

APP_FOLDER = Path(__file__).resolve().parent
DTM_PATH = APP_FOLDER / "data" / "here.tif"


# ============================================================
# 2. CUSTOM ERROR
# ============================================================

class TerrainSamplingError(RuntimeError):
    """Raised when a valid DTM elevation cannot be retrieved."""


# ============================================================
# 3. OPEN AND VALIDATE THE DTM
# ============================================================

@lru_cache(maxsize=1)
def get_dtm_dataset() -> DatasetReader:
    """
    Open the DTM once and reuse the same dataset during the app session.

    Rasterio reads only the requested pixels or windows, so the complete
    raster is not loaded into memory.
    """

    if not DTM_PATH.exists():
        raise TerrainSamplingError(
            f"DTM file was not found at: {DTM_PATH}"
        )

    try:
        dataset = rasterio.open(DTM_PATH)

    except Exception as error:
        raise TerrainSamplingError(
            f"Could not open the DTM GeoTIFF: {error}"
        ) from error

    if dataset.crs is None:
        dataset.close()
        raise TerrainSamplingError(
            "The DTM GeoTIFF does not have a coordinate system."
        )

    if dataset.count < 1:
        dataset.close()
        raise TerrainSamplingError(
            "The DTM GeoTIFF does not contain a raster band."
        )

    return dataset


@lru_cache(maxsize=1)
def get_wgs84_to_dtm_transformer() -> Transformer:
    """Create and cache the WGS84-to-DTM coordinate transformer."""

    dataset = get_dtm_dataset()

    return Transformer.from_crs(
        CRS.from_epsg(4326),
        dataset.crs,
        always_xy=True,
    )


# ============================================================
# 4. VALUE VALIDATION
# ============================================================

def _clean_elevation_value(
    value: float,
    nodata_value: float | None,
) -> float:
    """Validate and return one elevation value as metres."""

    elevation = float(value)

    if not np.isfinite(elevation):
        raise TerrainSamplingError(
            "The selected DTM cell does not contain a valid elevation."
        )

    if (
        nodata_value is not None
        and np.isclose(
            elevation,
            float(nodata_value),
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise TerrainSamplingError(
            "The selected point is outside the valid DTM area."
        )

    return elevation


# ============================================================
# 5. SAMPLE ONE LOCATION
# ============================================================

def sample_elevation_xy(
    x: float,
    y: float,
) -> float:
    """
    Sample the DTM at one coordinate in the raster's own CRS.

    Parameters
    ----------
    x:
        Easting or longitude in the raster CRS.
    y:
        Northing or latitude in the raster CRS.

    Returns
    -------
    float
        DTM elevation value in metres.
    """

    dataset = get_dtm_dataset()

    if not (
        dataset.bounds.left <= x <= dataset.bounds.right
        and dataset.bounds.bottom <= y <= dataset.bounds.top
    ):
        raise TerrainSamplingError(
            "The selected point is outside the DTM coverage."
        )

    try:
        sample = next(
            dataset.sample(
                [(float(x), float(y))],
                indexes=1,
                masked=True,
            )
        )

    except Exception as error:
        raise TerrainSamplingError(
            f"Could not sample the DTM: {error}"
        ) from error

    if np.ma.isMaskedArray(sample) and np.any(sample.mask):
        raise TerrainSamplingError(
            "The selected point falls on a DTM NoData cell."
        )

    return _clean_elevation_value(
        value=sample[0],
        nodata_value=dataset.nodata,
    )


def sample_elevation_wgs84(
    longitude: float,
    latitude: float,
) -> float:
    """
    Sample the DTM using a WGS84 longitude and latitude.

    Returns the ground elevation in metres.
    """

    transformer = get_wgs84_to_dtm_transformer()

    try:
        x, y = transformer.transform(
            float(longitude),
            float(latitude),
        )

    except Exception as error:
        raise TerrainSamplingError(
            f"Could not transform the coordinate to the DTM CRS: {error}"
        ) from error

    return sample_elevation_xy(
        x=float(x),
        y=float(y),
    )


# ============================================================
# 6. SAMPLE MANY PROJECTED LOCATIONS
# ============================================================

def sample_elevations_xy(
    coordinates: list[tuple[float, float]],
) -> list[float]:
    """
    Sample many coordinates in the raster CRS efficiently.

    Invalid or NoData samples are returned as NaN so a profile can still
    be displayed when only a small number of points are missing.
    """

    dataset = get_dtm_dataset()

    if not coordinates:
        return []

    try:
        sampled_values = dataset.sample(
            coordinates,
            indexes=1,
            masked=True,
        )

    except Exception as error:
        raise TerrainSamplingError(
            f"Could not sample the DTM profile: {error}"
        ) from error

    elevations: list[float] = []

    for sampled in sampled_values:
        try:
            if (
                np.ma.isMaskedArray(sampled)
                and np.any(sampled.mask)
            ):
                elevations.append(float("nan"))
                continue

            elevation = _clean_elevation_value(
                value=sampled[0],
                nodata_value=dataset.nodata,
            )

            elevations.append(elevation)

        except TerrainSamplingError:
            elevations.append(float("nan"))

    return elevations


# ============================================================
# 7. STRAIGHT-LINE ELEVATION PROFILE
# ============================================================

def sample_profile_wgs84(
    start_longitude: float,
    start_latitude: float,
    end_longitude: float,
    end_latitude: float,
    sample_count: int = 120,
) -> list[dict[str, float]]:
    """
    Sample a straight-line DTM elevation profile between two WGS84 points.

    Distance is calculated in the DTM's projected CRS and returned in metres.
    """

    if sample_count < 2:
        raise ValueError(
            "sample_count must be at least 2."
        )

    transformer = get_wgs84_to_dtm_transformer()

    try:
        start_x, start_y = transformer.transform(
            float(start_longitude),
            float(start_latitude),
        )

        end_x, end_y = transformer.transform(
            float(end_longitude),
            float(end_latitude),
        )

    except Exception as error:
        raise TerrainSamplingError(
            f"Could not transform profile coordinates: {error}"
        ) from error

    x_values = np.linspace(
        float(start_x),
        float(end_x),
        int(sample_count),
    )

    y_values = np.linspace(
        float(start_y),
        float(end_y),
        int(sample_count),
    )

    total_distance = float(
        np.hypot(
            float(end_x) - float(start_x),
            float(end_y) - float(start_y),
        )
    )

    distances = np.linspace(
        0.0,
        total_distance,
        int(sample_count),
    )

    coordinates = [
        (float(x), float(y))
        for x, y in zip(
            x_values,
            y_values,
        )
    ]

    elevations = sample_elevations_xy(
        coordinates
    )

    valid_count = sum(
        np.isfinite(value)
        for value in elevations
    )

    if valid_count < 2:
        raise TerrainSamplingError(
            "The terrain profile did not contain enough valid DTM values."
        )

    return [
        {
            "Distance (m)": float(distance),
            "Elevation (m)": float(elevation),
        }
        for distance, elevation in zip(
            distances,
            elevations,
        )
    ]


# ============================================================
# 8. OPTIONAL DTM INFORMATION
# ============================================================

def get_dtm_information() -> dict[str, object]:
    """Return basic DTM properties for diagnostics or display."""

    dataset = get_dtm_dataset()

    return {
        "path": str(DTM_PATH),
        "crs": str(dataset.crs),
        "width": int(dataset.width),
        "height": int(dataset.height),
        "resolution_x": float(dataset.res[0]),
        "resolution_y": float(abs(dataset.res[1])),
        "nodata": dataset.nodata,
        "data_type": dataset.dtypes[0],
        "bounds": {
            "left": float(dataset.bounds.left),
            "bottom": float(dataset.bounds.bottom),
            "right": float(dataset.bounds.right),
            "top": float(dataset.bounds.top),
        },
    }
