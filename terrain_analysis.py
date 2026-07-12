"""
Terrain Analysis module for the UiTM Shah Alam GIS dashboard.

This module displays the ArcGIS Online DTM and DSM tile services in a
separate Streamlit tab. It supports:
- DTM display
- DSM display
- DTM and DSM layer switching
- DTM versus DSM swipe comparison
- Orthophoto on/off
- Terrain opacity
- Building and road overlays
- Fullscreen and layer controls

The published MapServer layers are visual tile layers. Numeric pixel
elevation queries require the original GeoTIFF or an ImageServer service.
"""

from __future__ import annotations

from typing import Any, Optional

import folium
import streamlit as st
from branca.element import MacroElement
from folium.plugins import Fullscreen, MousePosition, SideBySideLayers
from jinja2 import Template
from streamlit_folium import st_folium


DTM_SERVICE_URL = (
    "https://tiles.arcgis.com/tiles/2ZRAaoTSJbQ20ceg/"
    "arcgis/rest/services/DTM_UiTM_Shah_Alam/MapServer"
)

DSM_SERVICE_URL = (
    "https://tiles.arcgis.com/tiles/2ZRAaoTSJbQ20ceg/"
    "arcgis/rest/services/DSM_UiTM_Shah_Alam/MapServer"
)

DTM_TILE_URL = f"{DTM_SERVICE_URL}/tile/{{z}}/{{y}}/{{x}}"
DSM_TILE_URL = f"{DSM_SERVICE_URL}/tile/{{z}}/{{y}}/{{x}}"


class TransparentWhiteTileLayer(MacroElement):
    """Remove near-white pixels from the orthophoto tiles."""

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

            var {{ this.get_name() }}Class = L.GridLayer.extend({
                createTile: function(coords, done) {
                    var tile = document.createElement("canvas");
                    var size = this.getTileSize();

                    tile.width = size.x;
                    tile.height = size.y;

                    var context = tile.getContext(
                        "2d",
                        {willReadFrequently: true}
                    );

                    var image = new Image();
                    image.crossOrigin = "anonymous";

                    var tileUrl = "{{ this.tile_url }}"
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

                            var imageData = context.getImageData(
                                0,
                                0,
                                size.x,
                                size.y
                            );

                            var pixels = imageData.data;
                            var threshold = {{ this.white_threshold }};

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

                                if (nearWhite && similarValues) {
                                    pixels[i + 3] = 0;
                                }
                            }

                            context.putImageData(imageData, 0, 0);
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
                    maxNativeZoom: {{ this.max_native_zoom }},
                    attribution: "UiTM Shah Alam UAV Orthomosaic"
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
    """Return candidate fields that exist in the GeoJSON properties."""

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


def _add_buildings(
    map_object: folium.Map,
    buildings_geojson: Optional[dict[str, Any]],
    show: bool,
) -> None:
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
        show=show,
        style_function=lambda feature: {
            "color": "#F39C12",
            "weight": 1.2,
            "fillColor": "#F39C12",
            "fillOpacity": 0.18,
        },
        highlight_function=lambda feature: {
            "color": "#FFD700",
            "weight": 3,
            "fillOpacity": 0.38,
        },
        tooltip=tooltip,
    ).add_to(map_object)


def _add_roads(
    map_object: folium.Map,
    roads_geojson: Optional[dict[str, Any]],
    show: bool,
) -> None:
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
        show=show,
        style_function=lambda feature: {
            "color": "#FFC0CB",
            "weight": 3,
            "opacity": 0.85,
        },
        tooltip=tooltip,
    ).add_to(map_object)


def create_terrain_map(
    display_mode: str,
    terrain_opacity: float,
    show_orthophoto: bool,
    show_buildings: bool,
    show_roads: bool,
    orthophoto_tile_url: Optional[str],
    buildings_geojson: Optional[dict[str, Any]],
    roads_geojson: Optional[dict[str, Any]],
    map_center: tuple[float, float],
    zoom_start: int,
) -> folium.Map:
    """Build the terrain visualisation map."""

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

    if orthophoto_tile_url and show_orthophoto:
        orthophoto_layer = TransparentWhiteTileLayer(
            tile_url=orthophoto_tile_url,
            white_threshold=245,
            opacity=1.0,
            max_native_zoom=20,
            max_zoom=23,
        )
        orthophoto_layer.add_to(terrain_map)

    if display_mode == "DTM":
        folium.TileLayer(
            tiles=DTM_TILE_URL,
            name="Digital Terrain Model",
            attr="UiTM Shah Alam DTM",
            overlay=True,
            control=True,
            show=True,
            opacity=terrain_opacity,
            max_native_zoom=20,
            max_zoom=23,
        ).add_to(terrain_map)

    elif display_mode == "DSM":
        folium.TileLayer(
            tiles=DSM_TILE_URL,
            name="Digital Surface Model",
            attr="UiTM Shah Alam DSM",
            overlay=True,
            control=True,
            show=True,
            opacity=terrain_opacity,
            max_native_zoom=20,
            max_zoom=23,
        ).add_to(terrain_map)

    elif display_mode == "DTM and DSM":
        folium.TileLayer(
            tiles=DTM_TILE_URL,
            name="Digital Terrain Model",
            attr="UiTM Shah Alam DTM",
            overlay=True,
            control=True,
            show=True,
            opacity=terrain_opacity,
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
            opacity=terrain_opacity,
            max_native_zoom=20,
            max_zoom=23,
        ).add_to(terrain_map)

    elif display_mode == "DTM vs DSM Swipe":
        dtm_layer = folium.TileLayer(
            tiles=DTM_TILE_URL,
            name="DTM — Left",
            attr="UiTM Shah Alam DTM",
            overlay=True,
            control=False,
            show=True,
            opacity=terrain_opacity,
            max_native_zoom=20,
            max_zoom=23,
        )

        dsm_layer = folium.TileLayer(
            tiles=DSM_TILE_URL,
            name="DSM — Right",
            attr="UiTM Shah Alam DSM",
            overlay=True,
            control=False,
            show=True,
            opacity=terrain_opacity,
            max_native_zoom=20,
            max_zoom=23,
        )

        dtm_layer.add_to(terrain_map)
        dsm_layer.add_to(terrain_map)

        SideBySideLayers(
            layer_left=dtm_layer,
            layer_right=dsm_layer,
        ).add_to(terrain_map)

    _add_roads(
        map_object=terrain_map,
        roads_geojson=roads_geojson,
        show=show_roads,
    )

    _add_buildings(
        map_object=terrain_map,
        buildings_geojson=buildings_geojson,
        show=show_buildings,
    )

    Fullscreen(
        position="topright",
        title="Full screen",
        title_cancel="Exit full screen",
        force_separate_button=True,
    ).add_to(terrain_map)

    MousePosition(
        position="bottomright",
        separator=" | ",
        prefix="Coordinate:",
        num_digits=6,
    ).add_to(terrain_map)

    layer_control = folium.LayerControl(
        collapsed=False,
        position="topleft",
    )
    layer_control.add_to(terrain_map)

    if orthophoto_layer is not None:
        terrain_map.get_root().script.add_child(
            folium.Element(
                f"""
                setTimeout(function() {{
                    try {{
                        {layer_control.get_name()}.addOverlay(
                            {orthophoto_layer.get_name()}_layer,
                            "UiTM Shah Alam Orthomosaic"
                        );
                    }} catch (error) {{
                        console.log(
                            "Orthomosaic control registration:",
                            error
                        );
                    }}
                }}, 500);
                """
            )
        )

    return terrain_map


def show_terrain_analysis(
    orthophoto_tile_url: Optional[str] = None,
    buildings_geojson: Optional[dict[str, Any]] = None,
    roads_geojson: Optional[dict[str, Any]] = None,
    map_center: tuple[float, float] = (3.0697, 101.5033),
    zoom_start: int = 16,
) -> None:
    """Render the complete terrain-analysis Streamlit interface."""

    st.subheader("⛰️ Terrain Analysis")

    st.caption(
        "Visualise and compare the Digital Terrain Model and Digital "
        "Surface Model generated from the UiTM Shah Alam UAV survey."
    )

    control_col1, control_col2 = st.columns([2, 1])

    with control_col1:
        display_mode = st.selectbox(
            "Terrain display",
            options=[
                "DTM",
                "DSM",
                "DTM and DSM",
                "DTM vs DSM Swipe",
            ],
            index=0,
            key="terrain_display_mode",
        )

    with control_col2:
        terrain_opacity = st.slider(
            "Terrain opacity",
            min_value=0.10,
            max_value=1.00,
            value=0.80,
            step=0.05,
            key="terrain_opacity",
        )

    toggle_col1, toggle_col2, toggle_col3 = st.columns(3)

    with toggle_col1:
        show_orthophoto = st.toggle(
            "Show orthophoto",
            value=True,
            key="terrain_show_orthophoto",
        )

    with toggle_col2:
        show_buildings = st.toggle(
            "Show buildings",
            value=False,
            key="terrain_show_buildings",
            disabled=buildings_geojson is None,
        )

    with toggle_col3:
        show_roads = st.toggle(
            "Show roads",
            value=False,
            key="terrain_show_roads",
            disabled=roads_geojson is None,
        )

    terrain_map = create_terrain_map(
        display_mode=display_mode,
        terrain_opacity=terrain_opacity,
        show_orthophoto=show_orthophoto,
        show_buildings=show_buildings,
        show_roads=show_roads,
        orthophoto_tile_url=orthophoto_tile_url,
        buildings_geojson=buildings_geojson,
        roads_geojson=roads_geojson,
        map_center=map_center,
        zoom_start=zoom_start,
    )

    map_result = st_folium(
        terrain_map,
        width=None,
        height=720,
        returned_objects=["last_clicked"],
        use_container_width=True,
        key=(
            f"terrain_map_{display_mode}_"
            f"{terrain_opacity}_{show_orthophoto}_"
            f"{show_buildings}_{show_roads}"
        ),
    )

    st.markdown("### Terrain information")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("Selected display", display_mode)

    with metric_col2:
        st.metric(
            "Terrain opacity",
            f"{terrain_opacity * 100:.0f}%",
        )

    with metric_col3:
        if display_mode == "DTM":
            representation = "Bare-earth terrain"
        elif display_mode == "DSM":
            representation = "Ground and objects"
        else:
            representation = "Surface comparison"

        st.metric("Representation", representation)

    clicked_point = map_result.get("last_clicked")

    if clicked_point:
        latitude = clicked_point.get("lat")
        longitude = clicked_point.get("lng")

        if latitude is not None and longitude is not None:
            st.success(
                "Selected coordinate: "
                f"{latitude:.6f}, {longitude:.6f}"
            )
    else:
        st.info("Click the map to inspect a terrain location.")

    with st.expander(
        "How to interpret DTM and DSM",
        expanded=False,
    ):
        st.markdown(
            """
            **Digital Terrain Model (DTM)**  
            Represents the approximate bare-earth ground surface. It is
            normally used for slope, aspect, drainage and terrain analysis.

            **Digital Surface Model (DSM)**  
            Represents the upper visible surface, including buildings,
            vegetation and other objects above the terrain.

            **DTM versus DSM swipe**  
            Drag the vertical slider to compare the bare-earth terrain with
            the surface containing above-ground features.
            """
        )

    st.warning(
        "These ArcGIS services are pre-rendered map tiles. They support "
        "visualisation and comparison, but not direct retrieval of the "
        "original elevation value. Slope, aspect, hillshade and DSM minus "
        "DTM should be generated from the source rasters and published as "
        "additional layers."
    )
