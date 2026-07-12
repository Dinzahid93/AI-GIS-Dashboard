"""
Terrain raster engine for the UiTM Shah Alam GIS dashboard.

Expected files inside the GitHub repository:

    data/here.tif
        Resampled DTM used for ground elevation.

    data/DSMUITM_Resample_CopyRast.tif
        Resampled DSM used for surface elevation and estimated object height.

This module supports:
- DTM ground elevation at one coordinate
- DSM surface elevation at one coordinate
- Estimated height at one coordinate: DSM - DTM
- Building-footprint statistics using DTM and DSM
- Straight-line DTM, DSM, or combined elevation profiles
- Automatic coordinate transformation from WGS84
- Windowed raster reading so the full raster is not loaded into memory

Compatibility:
The existing function sample_elevation_wgs84() is preserved and returns DTM
ground elevation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.io import DatasetReader
from rasterio.windows import Window, from_bounds
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform


RasterName = Literal["dtm", "dsm"]


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

APP_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = APP_FOLDER / "data"

DTM_PATH = DATA_FOLDER / "here.tif"


def _find_dsm_path() -> Path:
    """
    Find the DSM file while allowing common filename variations.

    The preferred filename is:
        data/DSMUITM_Resample_CopyRast.tif
    """

    candidates = [
        DATA_FOLDER / "DSMUITM_Resample_CopyRast.tif",
        DATA_FOLDER / "DSMUITM_Resample_CopyRast.TIF",
        DATA_FOLDER / "DSMUITM_Resample_CopyRast",
        DATA_FOLDER / "dsm.tif",
        DATA_FOLDER / "DSM.tif",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Return the preferred path so the error message is clear.
    return DATA_FOLDER / "DSMUITM_Resample_CopyRast.tif"


DSM_PATH = _find_dsm_path()


# ============================================================
# 2. CUSTOM ERROR
# ============================================================

class TerrainSamplingError(RuntimeError):
    """Raised when terrain sampling cannot return a valid value."""


# ============================================================
# 3. DATASET MANAGEMENT
# ============================================================

def _validate_dataset(
    dataset: DatasetReader,
    raster_label: str,
) -> DatasetReader:
    """Validate an opened GeoTIFF."""

    if dataset.crs is None:
        dataset.close()
        raise TerrainSamplingError(
            f"The {raster_label} GeoTIFF does not have a coordinate system."
        )

    if dataset.count < 1:
        dataset.close()
        raise TerrainSamplingError(
            f"The {raster_label} GeoTIFF does not contain a raster band."
        )

    return dataset


@lru_cache(maxsize=1)
def get_dtm_dataset() -> DatasetReader:
    """Open and cache the DTM dataset."""

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

    return _validate_dataset(
        dataset=dataset,
        raster_label="DTM",
    )


@lru_cache(maxsize=1)
def get_dsm_dataset() -> DatasetReader:
    """Open and cache the DSM dataset."""

    if not DSM_PATH.exists():
        raise TerrainSamplingError(
            "DSM file was not found. Expected one of these names inside "
            f"{DATA_FOLDER}: DSMUITM_Resample_CopyRast.tif, dsm.tif or DSM.tif."
        )

    try:
        dataset = rasterio.open(DSM_PATH)
    except Exception as error:
        raise TerrainSamplingError(
            f"Could not open the DSM GeoTIFF: {error}"
        ) from error

    return _validate_dataset(
        dataset=dataset,
        raster_label="DSM",
    )


def get_dataset(
    raster_name: RasterName,
) -> DatasetReader:
    """Return the requested raster dataset."""

    raster_key = raster_name.lower()

    if raster_key == "dtm":
        return get_dtm_dataset()

    if raster_key == "dsm":
        return get_dsm_dataset()

    raise ValueError(
        "raster_name must be 'dtm' or 'dsm'."
    )


# ============================================================
# 4. COORDINATE TRANSFORMERS
# ============================================================

@lru_cache(maxsize=2)
def get_wgs84_to_raster_transformer(
    raster_name: RasterName,
) -> Transformer:
    """Create and cache a WGS84-to-raster transformer."""

    dataset = get_dataset(raster_name)

    return Transformer.from_crs(
        CRS.from_epsg(4326),
        dataset.crs,
        always_xy=True,
    )


@lru_cache(maxsize=2)
def get_raster_to_wgs84_transformer(
    raster_name: RasterName,
) -> Transformer:
    """Create and cache a raster-to-WGS84 transformer."""

    dataset = get_dataset(raster_name)

    return Transformer.from_crs(
        dataset.crs,
        CRS.from_epsg(4326),
        always_xy=True,
    )


# ============================================================
# 5. VALUE VALIDATION
# ============================================================

def _clean_value(
    value: float,
    nodata_value: float | None,
    raster_label: str,
) -> float:
    """Validate one raster value."""

    numeric_value = float(value)

    if not np.isfinite(numeric_value):
        raise TerrainSamplingError(
            f"The selected {raster_label} cell does not contain a valid value."
        )

    if (
        nodata_value is not None
        and np.isclose(
            numeric_value,
            float(nodata_value),
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise TerrainSamplingError(
            f"The selected point falls on a {raster_label} NoData cell."
        )

    return numeric_value


def _is_inside_dataset(
    dataset: DatasetReader,
    x: float,
    y: float,
) -> bool:
    """Check whether a projected coordinate lies inside a raster."""

    return (
        dataset.bounds.left <= x <= dataset.bounds.right
        and dataset.bounds.bottom <= y <= dataset.bounds.top
    )


# ============================================================
# 6. SAMPLE ONE PROJECTED LOCATION
# ============================================================

def sample_raster_xy(
    x: float,
    y: float,
    raster_name: RasterName,
) -> float:
    """Sample DTM or DSM at one coordinate in that raster's CRS."""

    dataset = get_dataset(raster_name)
    raster_label = raster_name.upper()

    if not _is_inside_dataset(
        dataset=dataset,
        x=float(x),
        y=float(y),
    ):
        raise TerrainSamplingError(
            f"The selected point is outside the {raster_label} coverage."
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
            f"Could not sample the {raster_label}: {error}"
        ) from error

    if np.ma.isMaskedArray(sample) and np.any(sample.mask):
        raise TerrainSamplingError(
            f"The selected point falls on a {raster_label} NoData cell."
        )

    return _clean_value(
        value=sample[0],
        nodata_value=dataset.nodata,
        raster_label=raster_label,
    )


def sample_elevation_xy(
    x: float,
    y: float,
) -> float:
    """
    Backward-compatible DTM sampling function.

    Returns ground elevation in metres.
    """

    return sample_raster_xy(
        x=x,
        y=y,
        raster_name="dtm",
    )


def sample_surface_elevation_xy(
    x: float,
    y: float,
) -> float:
    """Return DSM surface elevation in metres."""

    return sample_raster_xy(
        x=x,
        y=y,
        raster_name="dsm",
    )


# ============================================================
# 7. SAMPLE ONE WGS84 LOCATION
# ============================================================

def sample_raster_wgs84(
    longitude: float,
    latitude: float,
    raster_name: RasterName,
) -> float:
    """Sample DTM or DSM using WGS84 longitude and latitude."""

    transformer = get_wgs84_to_raster_transformer(
        raster_name
    )

    try:
        x, y = transformer.transform(
            float(longitude),
            float(latitude),
        )
    except Exception as error:
        raise TerrainSamplingError(
            f"Could not transform the coordinate to the "
            f"{raster_name.upper()} CRS: {error}"
        ) from error

    return sample_raster_xy(
        x=float(x),
        y=float(y),
        raster_name=raster_name,
    )


def sample_elevation_wgs84(
    longitude: float,
    latitude: float,
) -> float:
    """
    Backward-compatible DTM function.

    Returns ground elevation in metres.
    """

    return sample_raster_wgs84(
        longitude=longitude,
        latitude=latitude,
        raster_name="dtm",
    )


def sample_surface_elevation_wgs84(
    longitude: float,
    latitude: float,
) -> float:
    """Return DSM surface elevation in metres."""

    return sample_raster_wgs84(
        longitude=longitude,
        latitude=latitude,
        raster_name="dsm",
    )


def sample_height_wgs84(
    longitude: float,
    latitude: float,
) -> dict[str, float]:
    """
    Return DTM, DSM and estimated height at one coordinate.

    Estimated height = DSM - DTM.
    """

    ground = sample_elevation_wgs84(
        longitude=longitude,
        latitude=latitude,
    )

    surface = sample_surface_elevation_wgs84(
        longitude=longitude,
        latitude=latitude,
    )

    estimated_height = surface - ground

    return {
        "ground_elevation_m": float(ground),
        "surface_elevation_m": float(surface),
        "estimated_height_m": float(estimated_height),
    }


# ============================================================
# 8. SAMPLE MANY PROJECTED LOCATIONS
# ============================================================

def sample_raster_values_xy(
    coordinates: list[tuple[float, float]],
    raster_name: RasterName,
) -> list[float]:
    """
    Sample many projected coordinates.

    Invalid or NoData values are returned as NaN.
    """

    dataset = get_dataset(raster_name)
    raster_label = raster_name.upper()

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
            f"Could not sample the {raster_label} profile: {error}"
        ) from error

    values: list[float] = []

    for sampled in sampled_values:
        try:
            if (
                np.ma.isMaskedArray(sampled)
                and np.any(sampled.mask)
            ):
                values.append(float("nan"))
                continue

            cleaned = _clean_value(
                value=sampled[0],
                nodata_value=dataset.nodata,
                raster_label=raster_label,
            )

            values.append(cleaned)

        except TerrainSamplingError:
            values.append(float("nan"))

    return values


def sample_elevations_xy(
    coordinates: list[tuple[float, float]],
) -> list[float]:
    """Backward-compatible multi-point DTM sampling."""

    return sample_raster_values_xy(
        coordinates=coordinates,
        raster_name="dtm",
    )


# ============================================================
# 9. BUILDING POLYGON STATISTICS
# ============================================================

def _transform_geometry_to_raster_crs(
    geometry_geojson: dict[str, Any],
    raster_name: RasterName,
):
    """Transform a WGS84 GeoJSON geometry into a raster CRS."""

    geometry = shape(geometry_geojson)

    transformer = get_wgs84_to_raster_transformer(
        raster_name
    )

    return shapely_transform(
        transformer.transform,
        geometry,
    )


def _read_polygon_values(
    geometry_geojson: dict[str, Any],
    raster_name: RasterName,
) -> np.ndarray:
    """
    Read valid raster pixels inside one WGS84 polygon.

    Only the polygon's raster window is read.
    """

    dataset = get_dataset(raster_name)
    raster_label = raster_name.upper()

    projected_geometry = _transform_geometry_to_raster_crs(
        geometry_geojson=geometry_geojson,
        raster_name=raster_name,
    )

    if projected_geometry.is_empty:
        raise TerrainSamplingError(
            f"The selected geometry is empty for {raster_label} analysis."
        )

    min_x, min_y, max_x, max_y = projected_geometry.bounds

    # Clip polygon bounds to raster bounds.
    clipped_left = max(min_x, dataset.bounds.left)
    clipped_bottom = max(min_y, dataset.bounds.bottom)
    clipped_right = min(max_x, dataset.bounds.right)
    clipped_top = min(max_y, dataset.bounds.top)

    if (
        clipped_left >= clipped_right
        or clipped_bottom >= clipped_top
    ):
        raise TerrainSamplingError(
            f"The building is outside the {raster_label} coverage."
        )

    try:
        window = from_bounds(
            clipped_left,
            clipped_bottom,
            clipped_right,
            clipped_top,
            transform=dataset.transform,
        )

        window = window.round_offsets().round_lengths()

        # Avoid zero-sized windows for very small polygons.
        if window.width < 1:
            window = Window(
                col_off=window.col_off,
                row_off=window.row_off,
                width=1,
                height=window.height,
            )

        if window.height < 1:
            window = Window(
                col_off=window.col_off,
                row_off=window.row_off,
                width=window.width,
                height=1,
            )

        raster_array = dataset.read(
            1,
            window=window,
            masked=True,
        )

        window_transform = dataset.window_transform(
            window
        )

        inside_mask = geometry_mask(
            [projected_geometry.__geo_interface__],
            out_shape=raster_array.shape,
            transform=window_transform,
            invert=True,
            all_touched=True,
        )

        combined_mask = np.logical_or(
            np.ma.getmaskarray(raster_array),
            np.logical_not(inside_mask),
        )

        values = np.ma.array(
            raster_array,
            mask=combined_mask,
        ).compressed()

    except Exception as error:
        raise TerrainSamplingError(
            f"Could not read {raster_label} values inside the building: "
            f"{error}"
        ) from error

    if values.size == 0:
        raise TerrainSamplingError(
            f"No valid {raster_label} pixels were found inside the building."
        )

    values = values.astype("float64")

    if dataset.nodata is not None:
        values = values[
            ~np.isclose(
                values,
                float(dataset.nodata),
                rtol=0.0,
                atol=1e-6,
            )
        ]

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        raise TerrainSamplingError(
            f"No valid {raster_label} values remained after NoData filtering."
        )

    return values


def summarize_polygon_raster(
    geometry_geojson: dict[str, Any],
    raster_name: RasterName,
) -> dict[str, float | int]:
    """Return useful statistics for DTM or DSM pixels inside a polygon."""

    values = _read_polygon_values(
        geometry_geojson=geometry_geojson,
        raster_name=raster_name,
    )

    return {
        "count": int(values.size),
        "minimum_m": float(np.min(values)),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "maximum_m": float(np.max(values)),
        "percentile_10_m": float(np.percentile(values, 10)),
        "percentile_90_m": float(np.percentile(values, 90)),
        "percentile_95_m": float(np.percentile(values, 95)),
    }


def calculate_building_height(
    building_geometry_geojson: dict[str, Any],
) -> dict[str, Any]:
    """
    Estimate building height using DTM and DSM pixels inside the footprint.

    Main values:
    - Ground elevation: median DTM within footprint
    - Surface elevation: median DSM within footprint
    - Estimated height: median DSM - median DTM

    Additional high-roof estimate:
    - 90th percentile DSM - median DTM

    Notes:
    The result is an estimated photogrammetric height. Trees or complex roof
    structures within the polygon may influence DSM statistics.
    """

    dtm_stats = summarize_polygon_raster(
        geometry_geojson=building_geometry_geojson,
        raster_name="dtm",
    )

    dsm_stats = summarize_polygon_raster(
        geometry_geojson=building_geometry_geojson,
        raster_name="dsm",
    )

    ground_median = float(
        dtm_stats["median_m"]
    )

    surface_median = float(
        dsm_stats["median_m"]
    )

    surface_p90 = float(
        dsm_stats["percentile_90_m"]
    )

    height_median = surface_median - ground_median
    height_p90 = surface_p90 - ground_median

    return {
        "ground_elevation_m": ground_median,
        "surface_elevation_m": surface_median,
        "estimated_height_m": float(height_median),
        "high_roof_height_m": float(height_p90),
        "dtm_statistics": dtm_stats,
        "dsm_statistics": dsm_stats,
    }


# ============================================================
# 10. STRAIGHT-LINE PROFILES
# ============================================================

def _profile_coordinates_in_raster_crs(
    start_longitude: float,
    start_latitude: float,
    end_longitude: float,
    end_latitude: float,
    sample_count: int,
    raster_name: RasterName,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Create profile coordinates and distances for one raster CRS."""

    transformer = get_wgs84_to_raster_transformer(
        raster_name
    )

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
            f"Could not transform profile coordinates to the "
            f"{raster_name.upper()} CRS: {error}"
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

    return coordinates, distances


def sample_profile_wgs84(
    start_longitude: float,
    start_latitude: float,
    end_longitude: float,
    end_latitude: float,
    sample_count: int = 120,
    surface: Literal["dtm", "dsm", "both"] = "dtm",
) -> list[dict[str, float]]:
    """
    Sample a straight-line terrain profile.

    surface:
        "dtm"  -> Distance and ground elevation
        "dsm"  -> Distance and surface elevation
        "both" -> Distance, DTM, DSM and DSM-DTM height
    """

    if sample_count < 2:
        raise ValueError(
            "sample_count must be at least 2."
        )

    if surface not in {
        "dtm",
        "dsm",
        "both",
    }:
        raise ValueError(
            "surface must be 'dtm', 'dsm' or 'both'."
        )

    reference_raster: RasterName = (
        "dtm"
        if surface in {"dtm", "both"}
        else "dsm"
    )

    coordinates, distances = (
        _profile_coordinates_in_raster_crs(
            start_longitude=start_longitude,
            start_latitude=start_latitude,
            end_longitude=end_longitude,
            end_latitude=end_latitude,
            sample_count=sample_count,
            raster_name=reference_raster,
        )
    )

    if surface == "dtm":
        dtm_values = sample_raster_values_xy(
            coordinates=coordinates,
            raster_name="dtm",
        )

        valid_count = sum(
            np.isfinite(value)
            for value in dtm_values
        )

        if valid_count < 2:
            raise TerrainSamplingError(
                "The DTM profile did not contain enough valid values."
            )

        return [
            {
                "Distance (m)": float(distance),
                "Elevation (m)": float(elevation),
            }
            for distance, elevation in zip(
                distances,
                dtm_values,
            )
        ]

    if surface == "dsm":
        dsm_values = sample_raster_values_xy(
            coordinates=coordinates,
            raster_name="dsm",
        )

        valid_count = sum(
            np.isfinite(value)
            for value in dsm_values
        )

        if valid_count < 2:
            raise TerrainSamplingError(
                "The DSM profile did not contain enough valid values."
            )

        return [
            {
                "Distance (m)": float(distance),
                "Surface elevation (m)": float(elevation),
            }
            for distance, elevation in zip(
                distances,
                dsm_values,
            )
        ]

    # Combined profile. DTM and DSM may use different CRS or alignment,
    # so sample each surface from the same WGS84 positions.
    fractions = np.linspace(
        0.0,
        1.0,
        int(sample_count),
    )

    longitudes = (
        float(start_longitude)
        + fractions
        * (
            float(end_longitude)
            - float(start_longitude)
        )
    )

    latitudes = (
        float(start_latitude)
        + fractions
        * (
            float(end_latitude)
            - float(start_latitude)
        )
    )

    dtm_values: list[float] = []
    dsm_values: list[float] = []

    for longitude, latitude in zip(
        longitudes,
        latitudes,
    ):
        try:
            dtm_value = sample_elevation_wgs84(
                longitude=float(longitude),
                latitude=float(latitude),
            )
        except TerrainSamplingError:
            dtm_value = float("nan")

        try:
            dsm_value = sample_surface_elevation_wgs84(
                longitude=float(longitude),
                latitude=float(latitude),
            )
        except TerrainSamplingError:
            dsm_value = float("nan")

        dtm_values.append(dtm_value)
        dsm_values.append(dsm_value)

    valid_pairs = sum(
        np.isfinite(dtm_value)
        and np.isfinite(dsm_value)
        for dtm_value, dsm_value in zip(
            dtm_values,
            dsm_values,
        )
    )

    if valid_pairs < 2:
        raise TerrainSamplingError(
            "The combined DTM/DSM profile did not contain enough valid values."
        )

    profile: list[dict[str, float]] = []

    for distance, dtm_value, dsm_value in zip(
        distances,
        dtm_values,
        dsm_values,
    ):
        height_value = (
            float(dsm_value - dtm_value)
            if (
                np.isfinite(dtm_value)
                and np.isfinite(dsm_value)
            )
            else float("nan")
        )

        profile.append(
            {
                "Distance (m)": float(distance),
                "Ground elevation (m)": float(dtm_value),
                "Surface elevation (m)": float(dsm_value),
                "Surface height (m)": height_value,
            }
        )

    return profile


# ============================================================
# 11. RASTER INFORMATION
# ============================================================

def _dataset_information(
    dataset: DatasetReader,
    path: Path,
) -> dict[str, object]:
    """Return reusable raster metadata."""

    return {
        "path": str(path),
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


def get_dtm_information() -> dict[str, object]:
    """Return DTM properties."""

    return _dataset_information(
        dataset=get_dtm_dataset(),
        path=DTM_PATH,
    )


def get_dsm_information() -> dict[str, object]:
    """Return DSM properties."""

    return _dataset_information(
        dataset=get_dsm_dataset(),
        path=DSM_PATH,
    )


def get_terrain_information() -> dict[str, object]:
    """Return both DTM and DSM properties."""

    return {
        "dtm": get_dtm_information(),
        "dsm": get_dsm_information(),
    }
