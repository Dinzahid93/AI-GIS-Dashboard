"""
Simplified Terrain Analysis module for the UiTM Shah Alam GIS dashboard.

This version keeps the interface clean:
- One terrain map only
- OpenStreetMap basemap
- Orthophoto overlay
- DTM overlay
- DSM overlay
- Layer control for switching layers on and off
- Fullscreen control
- No swipe comparison
- No opacity slider
- No building or road toggles
- No map-click feedback
- No interaction-based Streamlit reruns
"""

from __future__ import annotations

from typing import Any, Optional

import folium
import streamlit as st
from branca.element import MacroElement
from folium.plugins import Fullscreen
from jinja2 import Template
from streamlit_folium import st_folium


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
# 3. CREATE TERRAIN MAP
# ============================================================

def create_terrain_map(
    orthophoto_tile_url: Optional[str],
    buildings_geojson: Optional[dict[str, Any]],
    roads_geojson: Optional[dict[str, Any]],
    map_center: tuple[float, float],
    zoom_start: int,
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
    """
    Display the simplified Terrain Analysis tab.

    Building and road layers are included in the map layer control and
    remain hidden by default for a clean initial map.
    """

    st.subheader("⛰️ Terrain Analysis")

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
    )

    st_folium(
        terrain_map,
        width=None,
        height=720,
        returned_objects=[],
        use_container_width=True,
        key="terrain_analysis_map",
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
            Represents the approximate bare-earth ground surface.

            **Digital Surface Model (DSM)**  
            Represents the upper visible surface, including buildings,
            vegetation and other above-ground objects.
            """
        )
