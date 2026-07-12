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
    """Build the terrain visualisation or selectable swipe map."""

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

    # In normal terrain display, add the custom transparent orthophoto.
    # In swipe display, the selected orthophoto is inserted into the
    # SideBySideLayers control instead.
    if (
        display_mode != "Swipe comparison"
        and orthophoto_tile_url
        and show_orthophoto
    ):
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
        "Visualise the DTM and DSM, or select any two available raster "
        "layers for an interactive swipe comparison."
    )

    control_col1 = st.container()

    with control_col1:
        display_mode = st.selectbox(
            "Terrain display",
            options=[
                "DTM",
                "DSM",
                "DTM and DSM",
            ],
            index=0,
            key="terrain_display_mode",
        )

    terrain_opacity = 1.0


    show_orthophoto = True
    show_buildings = False
    show_roads = False


    try:
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

    except Exception as error:
        st.error(f"Could not prepare the terrain map: {error}")
        return

    st_folium(
        terrain_map,
        width=None,
        height=720,
        returned_objects=[],
        use_container_width=True,
        key=f"terrain_map_{display_mode}",
    )

    st.markdown("### Terrain information")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "Selected display",
            display_mode,
        )

    with metric_col2:
        st.metric(
            "Layer mode",
            "Single display",
        )

    with metric_col3:
        if display_mode == "DTM":
            representation = "Bare-earth terrain"
        elif display_mode == "DSM":
            representation = "Ground and objects"
        else:
            representation = "Surface comparison"

        st.metric("Representation", representation)

    with st.expander(
        "How to interpret DTM and DSM",
        expanded=False,
    ):
        st.markdown(
            """
            **Digital Terrain Model (DTM)**  
            Represents the approximate bare-earth ground surface.

            **Digital Surface Model (DSM)**  
            Represents the upper visible surface, including buildings,
            vegetation and other above-ground objects.

            **DTM and DSM**  
            Both layers are available in the map layer control. Switch
            between them to compare the ground and surface representations.
            """
        )

    st.warning(
        "These ArcGIS services are pre-rendered map tiles. They support "
        "visualisation and comparison, but not direct retrieval of the "
        "original elevation value. Slope, aspect, hillshade and DSM minus "
        "DTM should be generated from the source rasters and published as "
        "additional layers."
    )
