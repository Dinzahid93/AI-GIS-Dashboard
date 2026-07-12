"""
Simplified Terrain Analysis module for the UiTM Shah Alam GIS dashboard.

This version keeps the interface clean and adds:
- One terrain map only
- Orthophoto, DTM, DSM, buildings and roads
- Terrain GIS Assistant
- Actual DTM elevation from data/here.tif
- Automatic building highlight and zoom
- Elevation marker and popup
- No swipe comparison
- No opacity slider
- No map-click reruns
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import folium
import streamlit as st
from branca.element import MacroElement
from folium.plugins import Fullscreen
from jinja2 import Template
from shapely.geometry import shape
from streamlit_folium import st_folium

from terrain_engine import (
    TerrainSamplingError,
    sample_elevation_wgs84,
)


# ============================================================
# 1. ARCGIS ONLINE SERVICES
# ============================================================

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


# ============================================================
# 2. TRANSPARENT ORTHOPHOTO LAYER
# ============================================================

class TransparentWhiteTileLayer(MacroElement):
    """
    Display the orthophoto while removing near-white background pixels.

    This matches the orthophoto behaviour already used in the network
    analysis map.
    """

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



def _existing_fields(
    geojson_data: Optional[dict[str, Any]],
    candidates: list[str],
) -> list[str]:
    """Return candidate fields that exist in a GeoJSON layer."""

    if not geojson_data:
        return []

    features = geojson_data.get("features", [])

    if not features:
        return []

    available_fields: set[str] = set()

    for feature in features[:100]:
        properties = feature.get("properties", {}) or {}
        available_fields.update(properties.keys())

    return [
        field
        for field in candidates
        if field in available_fields
    ]



def _find_building_feature(
    buildings_geojson: dict[str, Any],
    building_id: int,
) -> dict[str, Any]:
    """Find one building feature using its FID."""

    for feature in buildings_geojson.get("features", []):
        properties = feature.get("properties", {}) or {}

        try:
            feature_id = int(properties.get("FID"))
        except (TypeError, ValueError):
            continue

        if feature_id == int(building_id):
            return feature

    raise ValueError(
        f"Building {building_id} was not found."
    )


def _building_point_wgs84(
    buildings_geojson: dict[str, Any],
    building_id: int,
) -> tuple[float, float]:
    """Return a representative longitude and latitude for a building."""

    feature = _find_building_feature(
        buildings_geojson=buildings_geojson,
        building_id=building_id,
    )

    geometry = shape(feature["geometry"])
    point = geometry.representative_point()

    return float(point.x), float(point.y)


def _building_name(
    buildings_geojson: dict[str, Any],
    building_id: int,
) -> str:
    """Return the building name or a fallback label."""

    feature = _find_building_feature(
        buildings_geojson=buildings_geojson,
        building_id=building_id,
    )

    properties = feature.get("properties", {}) or {}
    name = str(properties.get("NAME", "")).strip()

    return name if name else f"Building {building_id}"


def _add_building_layer(
    map_object: folium.Map,
    buildings_geojson: Optional[dict[str, Any]],
) -> None:
    """Add building footprints to the map layer control, hidden by default."""

    if not buildings_geojson:
        return

    tooltip_fields = _existing_fields(
        buildings_geojson,
        ["FID", "NAME", "name", "building_name"],
    )

    tooltip = None

    if tooltip_fields:
        tooltip = folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[
                field.replace("_", " ").title() + ":"
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
        highlight_function=lambda feature: {
            "color": "#FFD700",
            "weight": 3,
            "fillOpacity": 0.35,
        },
        tooltip=tooltip,
    ).add_to(map_object)


def _add_road_layer(
    map_object: folium.Map,
    roads_geojson: Optional[dict[str, Any]],
) -> None:
    """Add the road network to the map layer control, hidden by default."""

    if not roads_geojson:
        return

    tooltip_fields = _existing_fields(
        roads_geojson,
        ["FID", "NAME", "name", "road_name"],
    )

    tooltip = None

    if tooltip_fields:
        tooltip = folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[
                field.replace("_", " ").title() + ":"
                for field in tooltip_fields
            ],
            sticky=True,
        )

    folium.GeoJson(
        roads_geojson,
        name="Road Network",
        show=False,
        style_function=lambda feature: {
            "color": "#FFC0CB",
            "weight": 3,
            "opacity": 0.85,
        },
        tooltip=tooltip,
    ).add_to(map_object)



# ============================================================
# 3. TERRAIN COMMAND INTERPRETER
# ============================================================

def _fallback_parse_terrain_command(
    question: str,
) -> dict[str, Any]:
    """Parse a building-elevation request without Gemini."""

    text = question.lower().strip()

    building_match = re.search(
        r"building\s*(\d+)",
        text,
    )

    terrain_words = [
        "elevation",
        "height",
        "ground level",
        "terrain level",
        "altitude",
    ]

    if (
        building_match is not None
        and any(word in text for word in terrain_words)
    ):
        building_id = int(building_match.group(1))

        return {
            "action": "building_elevation",
            "building_id": building_id,
            "reply": (
                f"Retrieving the DTM ground elevation "
                f"for Building {building_id}."
            ),
        }

    return {
        "action": "unknown",
        "reply": (
            "Try: What is the elevation at Building 10?"
        ),
    }


def _extract_json_object(
    response_text: str,
) -> dict[str, Any]:
    """Extract one JSON object from a Gemini response."""

    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Gemini did not return a JSON object."
        )

    return json.loads(
        cleaned[start : end + 1]
    )


def _interpret_terrain_command(
    question: str,
    api_key: str,
    model_name: str,
) -> dict[str, Any]:
    """
    Use Gemini to interpret the request.

    The actual elevation is always calculated by the GIS engine.
    """

    if not api_key:
        return _fallback_parse_terrain_command(question)

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        prompt = f"""
You are a GIS command interpreter for a terrain dashboard.

The dashboard currently supports one action:
building_elevation

Return JSON only:
{{
  "action": "building_elevation",
  "building_id": 10,
  "reply": "Retrieving the DTM ground elevation for Building 10."
}}

Rules:
- Extract the building number exactly.
- Never invent an elevation value.
- For unsupported requests return:
{{
  "action": "unknown",
  "reply": "The terrain request is not supported."
}}

User request:
{question}
"""

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
            },
        )

        parsed = _extract_json_object(
            str(response.text)
        )

        if parsed.get("action") not in {
            "building_elevation",
            "unknown",
        }:
            raise ValueError(
                "Gemini returned an unsupported action."
            )

        return parsed

    except Exception:
        return _fallback_parse_terrain_command(question)


# ============================================================
# 4. CREATE TERRAIN MAP
# ============================================================

def create_terrain_map(
    orthophoto_tile_url: Optional[str],
    buildings_geojson: Optional[dict[str, Any]],
    roads_geojson: Optional[dict[str, Any]],
    map_center: tuple[float, float],
    zoom_start: int,
    selected_result: Optional[dict[str, Any]] = None,
) -> folium.Map:
    """
    Create one stable terrain map containing Orthophoto, DTM and DSM.

    Users control visibility directly from the Folium layer control.
    """

    terrain_map = folium.Map(
        location=list(map_center),
        zoom_start=zoom_start,
        tiles=None,
        max_zoom=23,
        control_scale=True,
        prefer_canvas=True,
    )

    # Base map
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
        show=True,
        max_zoom=23,
    ).add_to(terrain_map)

    # Orthophoto overlay
    orthophoto_layer = None

    if orthophoto_tile_url:
        orthophoto_layer = TransparentWhiteTileLayer(
            tile_url=orthophoto_tile_url,
            white_threshold=245,
            opacity=1.0,
            max_native_zoom=20,
            max_zoom=23,
        )

        orthophoto_layer.add_to(terrain_map)

    # DTM overlay
    folium.TileLayer(
        tiles=DTM_TILE_URL,
        name="Digital Terrain Model",
        attr="UiTM Shah Alam DTM",
        overlay=True,
        control=True,
        show=False,
        opacity=1.0,
        max_native_zoom=20,
        max_zoom=23,
    ).add_to(terrain_map)

    # DSM overlay
    folium.TileLayer(
        tiles=DSM_TILE_URL,
        name="Digital Surface Model",
        attr="UiTM Shah Alam DSM",
        overlay=True,
        control=True,
        show=False,
        opacity=1.0,
        max_native_zoom=20,
        max_zoom=23,
    ).add_to(terrain_map)

    # Optional vector overlays, available in the same layer control.
    _add_road_layer(
        map_object=terrain_map,
        roads_geojson=roads_geojson,
    )

    _add_building_layer(
        map_object=terrain_map,
        buildings_geojson=buildings_geojson,
    )

    # Highlight the building used in the latest elevation query.
    if (
        selected_result is not None
        and buildings_geojson is not None
    ):
        selected_building_id = int(
            selected_result["building_id"]
        )

        selected_feature = _find_building_feature(
            buildings_geojson=buildings_geojson,
            building_id=selected_building_id,
        )

        selected_geometry = shape(
            selected_feature["geometry"]
        )

        selected_point = (
            selected_geometry.representative_point()
        )

        selected_name = str(
            selected_result.get(
                "building_name",
                f"Building {selected_building_id}",
            )
        )

        selected_elevation = float(
            selected_result["elevation_m"]
        )

        selected_layer = folium.FeatureGroup(
            name="Selected Elevation Building",
            show=True,
        )

        folium.GeoJson(
            selected_feature,
            name=(
                f"Selected Building "
                f"{selected_building_id}"
            ),
            style_function=lambda feature: {
                "color": "#DC2626",
                "weight": 5,
                "fillColor": "#FDE047",
                "fillOpacity": 0.72,
            },
            highlight_function=lambda feature: {
                "color": "#991B1B",
                "weight": 6,
                "fillOpacity": 0.88,
            },
            tooltip=folium.Tooltip(
                f"<b>Building {selected_building_id}</b><br>"
                f"{selected_name}<br>"
                f"Ground elevation: "
                f"{selected_elevation:.2f} m",
                sticky=True,
            ),
        ).add_to(selected_layer)

        folium.Marker(
            location=[
                selected_point.y,
                selected_point.x,
            ],
            tooltip=(
                f"Building {selected_building_id}: "
                f"{selected_elevation:.2f} m"
            ),
            popup=folium.Popup(
                html=(
                    f"<b>Building {selected_building_id}</b><br>"
                    f"{selected_name}<br>"
                    f"<b>Ground elevation:</b> "
                    f"{selected_elevation:.2f} m"
                ),
                max_width=280,
            ),
            icon=folium.DivIcon(
                icon_size=(82, 34),
                icon_anchor=(41, 17),
                html=f"""
                <div style="
                    background:#DC2626;
                    color:#FFFFFF;
                    border:3px solid #FFFFFF;
                    border-radius:18px;
                    min-width:76px;
                    padding:5px 9px;
                    text-align:center;
                    font-size:13px;
                    font-weight:700;
                    box-shadow:0 2px 7px rgba(0,0,0,.55);
                    white-space:nowrap;
                ">
                    {selected_elevation:.2f} m
                </div>
                """,
            ),
        ).add_to(selected_layer)

        selected_layer.add_to(terrain_map)

        selected_bounds = selected_geometry.bounds

        terrain_map.fit_bounds(
            [
                [
                    selected_bounds[1],
                    selected_bounds[0],
                ],
                [
                    selected_bounds[3],
                    selected_bounds[2],
                ],
            ],
            padding=(120, 120),
            max_zoom=19,
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

    layer_control.add_to(terrain_map)

    # Register the custom orthophoto layer in the normal layer control.
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
                            "Orthophoto layer registration:",
                            error
                        );
                    }}
                }}, 500);
                """
            )
        )

    return terrain_map


# ============================================================
# 4. STREAMLIT TERRAIN TAB
# ============================================================

def show_terrain_analysis(
    orthophoto_tile_url: Optional[str] = None,
    buildings_geojson: Optional[dict[str, Any]] = None,
    roads_geojson: Optional[dict[str, Any]] = None,
    map_center: tuple[float, float] = (3.0697, 101.5033),
    zoom_start: int = 16,
) -> None:
    """Display the LLM-enabled Terrain Analysis tab."""

    if "terrain_elevation_result" not in st.session_state:
        st.session_state.terrain_elevation_result = None

    if "terrain_question" not in st.session_state:
        st.session_state.terrain_question = ""

    st.subheader("⛰️ Terrain Analysis")

    st.caption(
        "Use natural language to retrieve ground elevation from the DTM. "
        "The selected building is highlighted automatically on the map."
    )

    # --------------------------------------------------------
    # TERRAIN GIS ASSISTANT
    # --------------------------------------------------------

    st.markdown("### Terrain GIS Assistant")

    with st.form(
        key="terrain_gis_assistant_form",
        clear_on_submit=False,
    ):
        terrain_question = st.text_input(
            "Enter a terrain GIS request",
            value=st.session_state.terrain_question,
            placeholder=(
                "Example: What is the elevation at Building 10?"
            ),
        )

        with st.expander(
            "Example terrain command",
            expanded=False,
        ):
            st.code(
                "What is the elevation at Building 10?"
            )

        run_col, clear_col = st.columns([3, 1])

        with run_col:
            terrain_submitted = st.form_submit_button(
                "▶ Run Terrain Analysis",
                type="primary",
                use_container_width=True,
            )

        with clear_col:
            terrain_clear = st.form_submit_button(
                "🗑️ Clear",
                use_container_width=True,
            )

    if terrain_clear:
        st.session_state.terrain_elevation_result = None
        st.session_state.terrain_question = ""
        st.rerun()

    if terrain_submitted:
        st.session_state.terrain_question = terrain_question

        if not terrain_question.strip():
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

                parsed = _interpret_terrain_command(
                    question=terrain_question,
                    api_key=api_key,
                    model_name=model_name,
                )

                if parsed.get("action") == "building_elevation":
                    building_id = int(
                        parsed["building_id"]
                    )

                    longitude, latitude = _building_point_wgs84(
                        buildings_geojson=buildings_geojson,
                        building_id=building_id,
                    )

                    with st.spinner(
                        "Reading the DTM elevation value..."
                    ):
                        elevation = sample_elevation_wgs84(
                            longitude=longitude,
                            latitude=latitude,
                        )

                    st.session_state.terrain_elevation_result = {
                        "building_id": building_id,
                        "building_name": _building_name(
                            buildings_geojson,
                            building_id,
                        ),
                        "longitude": longitude,
                        "latitude": latitude,
                        "elevation_m": float(elevation),
                    }

                    st.success(
                        f"Building {building_id} was identified and "
                        "analysed successfully."
                    )

                else:
                    st.warning(
                        parsed.get(
                            "reply",
                            "The terrain request is not supported.",
                        )
                    )

            except TerrainSamplingError as error:
                st.error(
                    f"Could not retrieve elevation: {error}"
                )

            except ValueError as error:
                st.error(str(error))

            except Exception as error:
                st.error(
                    f"Terrain analysis failed: {error}"
                )

    # --------------------------------------------------------
    # RESULT CARD
    # --------------------------------------------------------

    result = st.session_state.terrain_elevation_result

    if result:
        st.markdown("### Terrain analysis result")

        metric_col1, metric_col2 = st.columns([2, 1])

        with metric_col1:
            st.metric(
                label=(
                    f"Ground elevation at "
                    f"Building {result['building_id']}"
                ),
                value=f"{result['elevation_m']:.2f} m",
            )

        with metric_col2:
            st.metric(
                label="Building ID",
                value=str(result["building_id"]),
            )

        st.caption(
            f"Building name: {result['building_name']}"
        )

        st.caption(
            "Sample coordinate: "
            f"{result['latitude']:.6f}, "
            f"{result['longitude']:.6f}"
        )

    # --------------------------------------------------------
    # TERRAIN MAP
    # --------------------------------------------------------

    st.markdown("### Terrain map")

    st.caption(
        "Use the map layer control to switch the orthophoto, DTM, DSM, "
        "buildings and road network on or off."
    )

    terrain_map = create_terrain_map(
        orthophoto_tile_url=orthophoto_tile_url,
        buildings_geojson=buildings_geojson,
        roads_geojson=roads_geojson,
        map_center=map_center,
        zoom_start=zoom_start,
        selected_result=result,
    )

    selected_id = (
        result["building_id"]
        if result
        else "none"
    )

    st_folium(
        terrain_map,
        width=None,
        height=720,
        returned_objects=[],
        use_container_width=True,
        key=f"terrain_analysis_map_{selected_id}",
    )

    with st.expander(
        "How to interpret the terrain layers",
        expanded=False,
    ):
        st.markdown(
            """
            **Orthophoto**  
            Shows the visible campus surface captured by the UAV camera.

            **Digital Terrain Model (DTM)**  
            Represents the approximate bare-earth ground surface. The
            elevation value is sampled from `data/here.tif` in metres.

            **Digital Surface Model (DSM)**  
            Represents the upper visible surface, including buildings,
            vegetation and other above-ground objects.
            """
        )
