"""
Terrain Analysis and Terrain GIS Assistant module.

Features:
- Orthophoto, DTM, DSM, roads and buildings in one map
- Natural-language terrain commands
- DTM elevation at a building
- Comparison of elevations between two buildings
- Straight-line terrain profile between two buildings
"""

from __future__ import annotations

from typing import Any, Optional

import folium
import requests
import numpy as np
import pandas as pd
import streamlit as st
from branca.element import MacroElement
from folium.plugins import Fullscreen
from jinja2 import Template
from shapely.geometry import shape
from streamlit_folium import st_folium

from terrain_ai import interpret_terrain_command
from terrain_engine import (
    TerrainSamplingError,
    sample_elevation_wgs84,
    sample_profile_wgs84,
)


DTM_SERVICE_URL = (
    "https://tiles.arcgis.com/tiles/2ZRAaoTSJbQ20ceg/"
    "arcgis/rest/services/DTM_UiTM_Shah_Alam/MapServer"
)

DSM_SERVICE_URL = (
    "https://tiles.arcgis.com/tiles/2ZRAaoTSJbQ20ceg/"
    "arcgis/rest/services/DSM_UiTM_Shah_Alam/MapServer"
)

DTM_TILE_URL = (
    f"{DTM_SERVICE_URL}/tile/{{z}}/{{y}}/{{x}}"
)

DSM_TILE_URL = (
    f"{DSM_SERVICE_URL}/tile/{{z}}/{{y}}/{{x}}"
)


class TransparentWhiteTileLayer(MacroElement):
    """Display orthophoto tiles while removing near-white NoData pixels."""

    def __init__(
        self,
        tile_url: str,
        white_threshold: int = 245,
        opacity: float = 1.0,
        max_native_zoom: int = 20,
        max_zoom: int = 23,
    ):
        super().__init__()

        self._name = "TransparentWhiteTileLayer"
        self.tile_url = tile_url
        self.white_threshold = white_threshold
        self.opacity = opacity
        self.max_native_zoom = max_native_zoom
        self.max_zoom = max_zoom

        self._template = Template(
            """
            {% macro script(this, kwargs) %}

            var {{ this.get_name() }}Class =
                L.GridLayer.extend({

                createTile: function(coords, done) {

                    var tile =
                        document.createElement("canvas");

                    var size = this.getTileSize();

                    tile.width = size.x;
                    tile.height = size.y;

                    var context = tile.getContext(
                        "2d",
                        {willReadFrequently: true}
                    );

                    var image = new Image();
                    image.crossOrigin = "anonymous";

                    var tileUrl =
                        "{{ this.tile_url }}"
                        .replace("{z}", coords.z)
                        .replace("{x}", coords.x)
                        .replace("{y}", coords.y);

                    image.onload = function() {

                        try {
                            context.drawImage(
                                image,
                                0,
                                0,
                                size.x,
                                size.y
                            );

                            var imageData =
                                context.getImageData(
                                    0,
                                    0,
                                    size.x,
                                    size.y
                                );

                            var pixels = imageData.data;
                            var threshold =
                                {{ this.white_threshold }};

                            for (
                                var i = 0;
                                i < pixels.length;
                                i += 4
                            ) {
                                var red = pixels[i];
                                var green = pixels[i + 1];
                                var blue = pixels[i + 2];

                                var nearWhite =
                                    red >= threshold &&
                                    green >= threshold &&
                                    blue >= threshold;

                                var similarValues =
                                    Math.abs(red - green) < 8 &&
                                    Math.abs(red - blue) < 8 &&
                                    Math.abs(green - blue) < 8;

                                if (
                                    nearWhite &&
                                    similarValues
                                ) {
                                    pixels[i + 3] = 0;
                                }
                            }

                            context.putImageData(
                                imageData,
                                0,
                                0
                            );

                            done(null, tile);

                        } catch (error) {
                            context.drawImage(
                                image,
                                0,
                                0,
                                size.x,
                                size.y
                            );

                            done(null, tile);
                        }
                    };

                    image.onerror = function(error) {
                        done(error, tile);
                    };

                    image.src = tileUrl;

                    return tile;
                }
            });

            var {{ this.get_name() }}_layer =
                new {{ this.get_name() }}Class({

                    tileSize: 256,
                    opacity: {{ this.opacity }},
                    maxZoom: {{ this.max_zoom }},
                    maxNativeZoom:
                        {{ this.max_native_zoom }},

                    attribution:
                        "UiTM Shah Alam UAV Orthomosaic"
                });

            {{ this.get_name() }}_layer.addTo(
                {{ this._parent.get_name() }}
            );

            {% endmacro %}
            """
        )


def _find_building_feature(
    buildings_geojson: dict[str, Any],
    building_id: int,
) -> dict[str, Any]:
    """Find one building feature by FID."""

    for feature in buildings_geojson.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        ) or {}

        try:
            fid = int(
                properties.get("FID")
            )
        except (TypeError, ValueError):
            continue

        if fid == int(building_id):
            return feature

    raise ValueError(
        f"Building {building_id} was not found."
    )


def _building_point(
    buildings_geojson: dict[str, Any],
    building_id: int,
) -> tuple[float, float]:
    """Return a representative longitude/latitude for one building."""

    feature = _find_building_feature(
        buildings_geojson=buildings_geojson,
        building_id=building_id,
    )

    geometry = shape(
        feature["geometry"]
    )

    point = geometry.representative_point()

    return float(point.x), float(point.y)


def _building_name(
    buildings_geojson: dict[str, Any],
    building_id: int,
) -> str:
    """Return the building name when available."""

    feature = _find_building_feature(
        buildings_geojson=buildings_geojson,
        building_id=building_id,
    )

    properties = feature.get(
        "properties",
        {},
    ) or {}

    name = str(
        properties.get("NAME", "")
    ).strip()

    return (
        name
        if name
        else f"Building {building_id}"
    )


def _existing_fields(
    geojson_data: Optional[dict[str, Any]],
    candidates: list[str],
) -> list[str]:
    if not geojson_data:
        return []

    features = geojson_data.get(
        "features",
        [],
    )

    if not features:
        return []

    fields: set[str] = set()

    for feature in features[:100]:
        fields.update(
            (feature.get("properties", {}) or {}).keys()
        )

    return [
        field
        for field in candidates
        if field in fields
    ]


def _add_vector_layers(
    map_object: folium.Map,
    buildings_geojson: Optional[dict[str, Any]],
    roads_geojson: Optional[dict[str, Any]],
) -> None:
    """Add road and building overlays, hidden by default."""

    if roads_geojson:
        folium.GeoJson(
            roads_geojson,
            name="Road Network",
            show=False,
            style_function=lambda feature: {
                "color": "#FFC0CB",
                "weight": 3,
                "opacity": 0.85,
            },
        ).add_to(map_object)

    if buildings_geojson:
        tooltip_fields = _existing_fields(
            buildings_geojson,
            ["FID", "NAME"],
        )

        tooltip = None

        if tooltip_fields:
            tooltip = folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=[
                    "Building ID:"
                    if field == "FID"
                    else "Building Name:"
                    for field in tooltip_fields
                ],
                sticky=True,
            )

        folium.GeoJson(
            buildings_geojson,
            name="Building Footprints",
            show=False,
            style_function=lambda feature: {
                "color": "#F39C12",
                "weight": 1.2,
                "fillColor": "#F39C12",
                "fillOpacity": 0.18,
            },
            tooltip=tooltip,
        ).add_to(map_object)


def create_terrain_map(
    orthophoto_tile_url: Optional[str],
    buildings_geojson: Optional[dict[str, Any]],
    roads_geojson: Optional[dict[str, Any]],
    map_center: tuple[float, float],
    zoom_start: int,
) -> folium.Map:
    """Create one stable terrain map."""

    terrain_map = folium.Map(
        location=list(map_center),
        zoom_start=zoom_start,
        tiles=None,
        max_zoom=23,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
        show=True,
        max_zoom=23,
    ).add_to(terrain_map)

    orthophoto_layer = None

    if orthophoto_tile_url:
        orthophoto_layer = TransparentWhiteTileLayer(
            tile_url=orthophoto_tile_url,
            white_threshold=245,
            opacity=1.0,
            max_native_zoom=20,
            max_zoom=23,
        )

        orthophoto_layer.add_to(
            terrain_map
        )

    folium.TileLayer(
        tiles=DTM_TILE_URL,
        name="Digital Terrain Model",
        attr="UiTM Shah Alam DTM",
        overlay=True,
        control=True,
        show=False,
        max_native_zoom=20,
        max_zoom=23,
    ).add_to(terrain_map)

    folium.TileLayer(
        tiles=DSM_TILE_URL,
        name="Digital Surface Model",
        attr="UiTM Shah Alam DSM",
        overlay=True,
        control=True,
        show=False,
        max_native_zoom=20,
        max_zoom=23,
    ).add_to(terrain_map)

    _add_vector_layers(
        map_object=terrain_map,
        buildings_geojson=buildings_geojson,
        roads_geojson=roads_geojson,
    )

    Fullscreen(
        position="topright",
        title="Full screen",
        title_cancel="Exit full screen",
        force_separate_button=True,
    ).add_to(terrain_map)

    layer_control = folium.LayerControl(
        collapsed=False,
        position="topleft",
    )

    layer_control.add_to(
        terrain_map
    )

    if orthophoto_layer is not None:
        terrain_map.get_root().script.add_child(
            folium.Element(
                f"""
                setTimeout(function() {{
                    try {{
                        {layer_control.get_name()}.addOverlay(
                            {orthophoto_layer.get_name()}_layer,
                            "UiTM Shah Alam Orthophoto"
                        );
                    }} catch (error) {{
                        console.log(
                            "Orthophoto registration:",
                            error
                        );
                    }}
                }}, 500);
                """
            )
        )

    return terrain_map


def _run_terrain_action(
    parsed: dict[str, Any],
    buildings_geojson: dict[str, Any],
) -> None:
    """Execute one structured terrain GIS action."""

    action = parsed.get(
        "action",
        "unknown",
    )

    reply = str(
        parsed.get("reply", "")
    ).strip()

    if reply:
        st.info(reply)

    if action == "building_elevation":
        building_id = int(
            parsed["building_id"]
        )

        longitude, latitude = _building_point(
            buildings_geojson,
            building_id,
        )

        elevation = sample_elevation_wgs84(
            longitude=longitude,
            latitude=latitude,
        )

        st.session_state.terrain_result = {
            "action": action,
            "building_id": building_id,
            "elevation": elevation,
        }

    elif action == "compare_building_elevations":
        building_ids = [
            int(value)
            for value in parsed["building_ids"]
        ][:2]

        if len(building_ids) != 2:
            raise ValueError(
                "Two building IDs are required."
            )

        results = []

        for building_id in building_ids:
            longitude, latitude = _building_point(
                buildings_geojson,
                building_id,
            )

            elevation = sample_elevation_wgs84(
                longitude=longitude,
                latitude=latitude,
            )

            results.append(
                {
                    "building_id": building_id,
                    "elevation": elevation,
                }
            )

        st.session_state.terrain_result = {
            "action": action,
            "results": results,
        }

    elif action == "elevation_profile":
        origin_id = int(
            parsed["origin_id"]
        )

        destination_id = int(
            parsed["destination_id"]
        )

        start_lon, start_lat = _building_point(
            buildings_geojson,
            origin_id,
        )

        end_lon, end_lat = _building_point(
            buildings_geojson,
            destination_id,
        )

        profile = sample_profile_wgs84(
            start_longitude=start_lon,
            start_latitude=start_lat,
            end_longitude=end_lon,
            end_latitude=end_lat,
            sample_count=120,
        )

        st.session_state.terrain_result = {
            "action": action,
            "origin_id": origin_id,
            "destination_id": destination_id,
            "profile": profile,
        }

    else:
        st.warning(
            "The terrain request was not recognised."
        )


def _display_terrain_result(
    buildings_geojson: dict[str, Any],
) -> None:
    """Display the current terrain GIS result."""

    result = st.session_state.get(
        "terrain_result"
    )

    if not result:
        return

    action = result["action"]

    st.markdown("### Terrain analysis result")

    if action == "building_elevation":
        building_id = result["building_id"]
        elevation = float(
            result["elevation"]
        )

        st.metric(
            f"DTM elevation at Building {building_id}",
            f"{elevation:.2f} m",
        )

        st.caption(
            _building_name(
                buildings_geojson,
                building_id,
            )
        )

    elif action == "compare_building_elevations":
        rows = []

        for item in result["results"]:
            building_id = item["building_id"]

            rows.append(
                {
                    "Building ID": building_id,
                    "Building name": _building_name(
                        buildings_geojson,
                        building_id,
                    ),
                    "DTM elevation (m)": round(
                        float(item["elevation"]),
                        2,
                    ),
                }
            )

        table = pd.DataFrame(rows)

        difference = abs(
            table.iloc[0]["DTM elevation (m)"]
            - table.iloc[1]["DTM elevation (m)"]
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        st.metric(
            "Elevation difference",
            f"{difference:.2f} m",
        )

    elif action == "elevation_profile":
        profile_table = pd.DataFrame(
            result["profile"]
        )

        valid = profile_table.dropna(
            subset=["Elevation (m)"]
        )

        origin_id = result["origin_id"]
        destination_id = result["destination_id"]

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Profile distance",
                f"{valid['Distance (m)'].max():.1f} m",
            )

        with metric2:
            st.metric(
                "Minimum elevation",
                f"{valid['Elevation (m)'].min():.2f} m",
            )

        with metric3:
            st.metric(
                "Maximum elevation",
                f"{valid['Elevation (m)'].max():.2f} m",
            )

        st.line_chart(
            profile_table,
            x="Distance (m)",
            y="Elevation (m)",
            use_container_width=True,
        )

        st.caption(
            "Straight-line DTM profile between "
            f"Building {origin_id} and Building {destination_id}. "
            "This is not yet sampled along the road route."
        )

        st.download_button(
            "Download elevation profile CSV",
            data=profile_table.to_csv(
                index=False,
            ).encode("utf-8-sig"),
            file_name=(
                f"dtm_profile_building_{origin_id}_"
                f"to_{destination_id}.csv"
            ),
            mime="text/csv",
        )


def show_terrain_analysis(
    orthophoto_tile_url: Optional[str] = None,
    buildings_geojson: Optional[dict[str, Any]] = None,
    roads_geojson: Optional[dict[str, Any]] = None,
    map_center: tuple[float, float] = (3.0697, 101.5033),
    zoom_start: int = 16,
) -> None:
    """Render the complete Terrain Analysis tab."""

    if "terrain_result" not in st.session_state:
        st.session_state.terrain_result = None

    st.subheader("⛰️ Terrain Analysis")

    st.caption(
        "Use natural-language commands to retrieve DTM elevation values "
        "and generate terrain profiles."
    )

    st.markdown("### Terrain GIS Assistant")

    with st.form(
        "terrain_gis_assistant_form",
        clear_on_submit=False,
    ):
        question = st.text_input(
            "Enter a terrain GIS request",
            placeholder=(
                "Example: Show the elevation profile "
                "from Building 10 to Building 20"
            ),
        )

        with st.expander(
            "Example terrain commands",
            expanded=False,
        ):
            st.markdown(
                """
                `What is the elevation at Building 10?`

                `Compare the elevation of Building 10 and Building 20.`

                `Show the elevation profile from Building 10 to Building 20.`
                """
            )

        run_col, clear_col = st.columns(
            [3, 1]
        )

        with run_col:
            submitted = st.form_submit_button(
                "▶ Run Terrain Analysis",
                type="primary",
                use_container_width=True,
            )

        with clear_col:
            clear_pressed = st.form_submit_button(
                "🗑️ Clear",
                use_container_width=True,
            )

    if clear_pressed:
        st.session_state.terrain_result = None
        st.rerun()

    if submitted:
        if not question.strip():
            st.warning(
                "Please enter a terrain GIS request."
            )

        elif buildings_geojson is None:
            st.error(
                "Building data are unavailable."
            )

        else:
            try:
                api_key = str(
                    st.secrets.get(
                        "GEMINI_API_KEY",
                        "",
                    )
                ).strip()

                model_name = str(
                    st.secrets.get(
                        "GEMINI_MODEL",
                        "gemini-2.5-flash-lite",
                    )
                ).strip()

                parsed = interpret_terrain_command(
                    question=question,
                    api_key=api_key,
                    model_name=model_name,
                )

                with st.spinner(
                    "Sampling the DTM elevation service..."
                ):
                    _run_terrain_action(
                        parsed=parsed,
                        buildings_geojson=buildings_geojson,
                    )

            except (
                TerrainSamplingError,
                ValueError,
                KeyError,
                requests.RequestException,
            ) as error:
                st.error(
                    f"Terrain analysis failed: {error}"
                )

            except Exception as error:
                st.error(
                    f"Unexpected terrain-analysis error: {error}"
                )

    if buildings_geojson is not None:
        _display_terrain_result(
            buildings_geojson
        )

    st.markdown("### Terrain map")

    st.caption(
        "Use the map layer control to switch the orthophoto, DTM, DSM, "
        "building footprints and road network on or off."
    )

    terrain_map = create_terrain_map(
        orthophoto_tile_url=orthophoto_tile_url,
        buildings_geojson=buildings_geojson,
        roads_geojson=roads_geojson,
        map_center=map_center,
        zoom_start=zoom_start,
    )

    st_folium(
        terrain_map,
        width=None,
        height=720,
        returned_objects=[],
        use_container_width=True,
        key="terrain_analysis_map",
    )
