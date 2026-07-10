from pathlib import Path

import folium
import geopandas as gpd
import streamlit as st
from branca.element import MacroElement
from folium.plugins import Fullscreen
from jinja2 import Template
from streamlit_folium import st_folium


# ============================================================
# 1. STREAMLIT PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI GIS Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    iframe {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. FILE AND WEB-LAYER SETTINGS
# ============================================================

APP_FOLDER = Path(__file__).resolve().parent

BUILDING_FILE = APP_FOLDER / "Building_FeaturesToJSON.geojson"
ROAD_FILE = APP_FOLDER / "ROAD_FeaturesToJSON.geojson"

ORTHO_TILE_URL = (
    "https://tiles.arcgis.com/tiles/2ZRAaoTSJbQ20ceg/"
    "arcgis/rest/services/UiTM_Shah_Alam_Orthomosaic/"
    "MapServer/tile/{z}/{y}/{x}"
)

BUILDING_COLOUR = "#F39C12"
ROAD_COLOUR = "#8B4513"


# ============================================================
# 4. LOAD GIS DATA
# ============================================================

@st.cache_data(show_spinner="Loading campus GIS layers...")
def load_gis_data():
    if not BUILDING_FILE.exists():
        raise FileNotFoundError(
            f"Building file not found: {BUILDING_FILE.name}"
        )

    if not ROAD_FILE.exists():
        raise FileNotFoundError(
            f"Road file not found: {ROAD_FILE.name}"
        )

    buildings = gpd.read_file(BUILDING_FILE)
    roads = gpd.read_file(ROAD_FILE)

    if buildings.crs is None:
        raise ValueError("Building layer has no coordinate system.")

    if roads.crs is None:
        raise ValueError("Road layer has no coordinate system.")

    if "FID" not in buildings.columns:
        raise ValueError("Building layer does not contain an FID field.")

    buildings = buildings.to_crs(epsg=4326)
    roads = roads.to_crs(epsg=4326)

    # Remove invalid or empty features
    buildings = buildings[
        buildings.geometry.notna()
        & ~buildings.geometry.is_empty
    ].copy()

    roads = roads[
        roads.geometry.notna()
        & ~roads.geometry.is_empty
    ].copy()

    return buildings, roads


try:
    buildings_wgs, roads_wgs = load_gis_data()

except Exception as error:
    st.error(f"Unable to load GIS data: {error}")
    st.stop()


# ============================================================
# 5. CUSTOM ORTHOMOSAIC TILE LAYER
#    Converts near-white JPEG pixels to transparent pixels.
# ============================================================

class TransparentWhiteTileLayer(MacroElement):
    """
    Leaflet GridLayer that downloads ArcGIS JPEG tiles, draws them
    onto a canvas, and converts near-white pixels to transparent.

    This allows OpenStreetMap underneath the orthomosaic to remain
    visible wherever the orthomosaic contains white NoData pixels.
    """

    def __init__(
        self,
        tile_url,
        layer_name="UiTM Shah Alam Orthomosaic",
        show=True,
        opacity=1.0,
        white_threshold=245,
        max_native_zoom=20,
        max_zoom=23,
    ):
        super().__init__()

        self._name = "TransparentWhiteTileLayer"
        self.tile_url = tile_url
        self.layer_name = layer_name
        self.show = show
        self.opacity = opacity
        self.white_threshold = white_threshold
        self.max_native_zoom = max_native_zoom
        self.max_zoom = max_zoom

        self._template = Template(
            """
            {% macro script(this, kwargs) %}

            var {{ this.get_name() }} = L.GridLayer.extend({

                createTile: function(coords, done) {

                    var tile = document.createElement("canvas");
                    var tileSize = this.getTileSize();

                    tile.width = tileSize.x;
                    tile.height = tileSize.y;

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
                                tileSize.x,
                                tileSize.y
                            );

                            var imageData = context.getImageData(
                                0,
                                0,
                                tileSize.x,
                                tileSize.y
                            );

                            var pixels = imageData.data;
                            var threshold = {{ this.white_threshold }};

                            for (var i = 0; i < pixels.length; i += 4) {

                                var red = pixels[i];
                                var green = pixels[i + 1];
                                var blue = pixels[i + 2];

                                /*
                                Make near-white pixels transparent.
                                Additional condition keeps pale real-world
                                surfaces from being removed unnecessarily.
                                */
                                var almostWhite =
                                    red >= threshold &&
                                    green >= threshold &&
                                    blue >= threshold;

                                var lowColourDifference =
                                    Math.abs(red - green) < 8 &&
                                    Math.abs(red - blue) < 8 &&
                                    Math.abs(green - blue) < 8;

                                if (almostWhite && lowColourDifference) {
                                    pixels[i + 3] = 0;
                                }
                            }

                            context.putImageData(imageData, 0, 0);
                            done(null, tile);

                        } catch (error) {
                            console.error(
                                "Orthomosaic transparency error:",
                                error
                            );

                            /*
                            Fallback: display original tile if browser
                            pixel processing is blocked.
                            */
                            context.clearRect(
                                0,
                                0,
                                tileSize.x,
                                tileSize.y
                            );

                            context.drawImage(
                                image,
                                0,
                                0,
                                tileSize.x,
                                tileSize.y
                            );

                            done(null, tile);
                        }
                    };

                    image.onerror = function(error) {
                        console.error(
                            "Unable to load orthomosaic tile:",
                            tileUrl
                        );
                        done(error, tile);
                    };

                    image.src = tileUrl;

                    return tile;
                }

            });

            var {{ this.get_name() }}_layer =
                new {{ this.get_name() }}({

                    tileSize: 256,
                    opacity: {{ this.opacity }},
                    minZoom: 0,
                    maxZoom: {{ this.max_zoom }},
                    maxNativeZoom: {{ this.max_native_zoom }},

                    attribution:
                        "UiTM Shah Alam UAV Orthomosaic"
                });

            {{ this._parent.get_name() }}.addLayer(
                {{ this.get_name() }}_layer
            );

            {% if this.show %}
            {{ this.get_name() }}_layer.addTo(
                {{ this._parent.get_name() }}
            );
            {% endif %}

            {% endmacro %}
            """
        )


# ============================================================
# 6. MAP CONTROL CSS
# ============================================================

class MapControlCSS(MacroElement):
    def __init__(self):
        super().__init__()

        self._template = Template(
            """
            {% macro html(this, kwargs) %}
            <style>
            .leaflet-control-layers {
                border-radius: 8px !important;
                box-shadow: 0 1px 6px rgba(0,0,0,0.25) !important;
                font-size: 14px !important;
                max-height: 320px !important;
                overflow-y: auto !important;
            }

            .leaflet-control-layers-expanded {
                padding: 9px 12px !important;
                background: rgba(255,255,255,0.95) !important;
            }

            .leaflet-control-zoom a,
            .leaflet-control-fullscreen a {
                border-radius: 4px !important;
            }
            </style>
            {% endmacro %}
            """
        )


# ============================================================
# 7. CREATE CAMPUS MAP
# ============================================================

def create_campus_map():
    campus_bounds = buildings_wgs.total_bounds
    min_x, min_y, max_x, max_y = campus_bounds

    centre_latitude = (min_y + max_y) / 2
    centre_longitude = (min_x + max_x) / 2

    campus_map = folium.Map(
        location=[centre_latitude, centre_longitude],
        zoom_start=16,
        tiles=None,
        max_zoom=23,
        control_scale=True,
        prefer_canvas=True,
    )

    # --------------------------------------------------------
    # OpenStreetMap stays underneath the orthomosaic.
    # --------------------------------------------------------

    openstreetmap = folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
        show=True,
        max_zoom=23,
    )

    openstreetmap.add_to(campus_map)

    # --------------------------------------------------------
    # Orthomosaic is an overlay with white NoData removed.
    # --------------------------------------------------------

    transparent_ortho = TransparentWhiteTileLayer(
        tile_url=ORTHO_TILE_URL,
        layer_name="UiTM Shah Alam Orthomosaic",
        show=True,
        opacity=1.0,
        white_threshold=245,
        max_native_zoom=20,
        max_zoom=23,
    )

    transparent_ortho.add_to(campus_map)

    # Register custom orthomosaic in Leaflet's overlay control
    campus_map.get_root().script.add_child(
        folium.Element(
            f"""
            <script>
            document.addEventListener("DOMContentLoaded", function() {{

                var map = {campus_map.get_name()};

                setTimeout(function() {{

                    var orthoLayer =
                        {transparent_ortho.get_name()}_layer;

                    if (
                        map &&
                        orthoLayer &&
                        map._controlLayers
                    ) {{
                        map._controlLayers.addOverlay(
                            orthoLayer,
                            "UiTM Shah Alam Orthomosaic"
                        );
                    }}

                }}, 500);

            }});
            </script>
            """
        )
    )

    # --------------------------------------------------------
    # Road network
    # --------------------------------------------------------

    road_layer = folium.GeoJson(
        roads_wgs,
        name="Road Network",
        style_function=lambda feature: {
            "color": ROAD_COLOUR,
            "weight": 3,
            "opacity": 0.85,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["FID"],
            aliases=["Road feature:"],
            sticky=True,
        )
        if "FID" in roads_wgs.columns
        else None,
    )

    road_layer.add_to(campus_map)

    # --------------------------------------------------------
    # Building footprints
    # --------------------------------------------------------

    building_layer = folium.GeoJson(
        buildings_wgs,
        name="Building Footprints",
        style_function=lambda feature: {
            "color": BUILDING_COLOUR,
            "weight": 1.5,
            "fillColor": BUILDING_COLOUR,
            "fillOpacity": 0.22,
        },
        highlight_function=lambda feature: {
            "color": "#FFD700",
            "weight": 3,
            "fillOpacity": 0.42,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["FID"],
            aliases=["Building ID:"],
            sticky=True,
        ),
    )

    building_layer.add_to(campus_map)

    # --------------------------------------------------------
    # Fullscreen control
    # --------------------------------------------------------

    Fullscreen(
        position="topright",
        title="Full screen",
        title_cancel="Exit full screen",
        force_separate_button=True,
    ).add_to(campus_map)

    # --------------------------------------------------------
    # Fit map around campus extent
    # --------------------------------------------------------

    campus_map.fit_bounds(
        [
            [min_y, min_x],
            [max_y, max_x],
        ],
        padding=(20, 20),
    )

    # --------------------------------------------------------
    # Layer control
    # --------------------------------------------------------

    folium.LayerControl(
        collapsed=False,
        position="topleft",
    ).add_to(campus_map)

    campus_map.get_root().add_child(MapControlCSS())

    return campus_map


# ============================================================
# 8. STREAMLIT INTERFACE
# ============================================================

st.title("🗺️ AI GIS Dashboard")

st.caption(
    "Interactive UiTM Shah Alam campus map using UAV orthomosaic, "
    "building footprints and road network."
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        label="Buildings",
        value=f"{len(buildings_wgs):,}",
    )

with metric2:
    st.metric(
        label="Road features",
        value=f"{len(roads_wgs):,}",
    )

with metric3:
    st.metric(
        label="Current capability",
        value="Map viewer",
    )

st.info(
    "The orthomosaic is displayed above OpenStreetMap. "
    "White NoData pixels are made transparent so the surrounding "
    "OpenStreetMap remains visible."
)

campus_map = create_campus_map()

st_folium(
    campus_map,
    width=None,
    height=720,
    returned_objects=[],
    use_container_width=True,
)
