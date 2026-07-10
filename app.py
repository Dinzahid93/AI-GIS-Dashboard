from pathlib import Path
import re

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from branca.element import MacroElement
from folium.plugins import Fullscreen
from jinja2 import Template
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from streamlit_folium import st_folium


# ============================================================
# 1. STREAMLIT PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI GIS Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    iframe {
        border-radius: 12px;
    }

    .route-summary {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. PROJECT FILES AND SETTINGS
# ============================================================

APP_FOLDER = Path(__file__).resolve().parent

BUILDING_FILE = APP_FOLDER / "Building_FeaturesToJSON.geojson"
ROAD_FILE = APP_FOLDER / "ROAD_FeaturesToJSON.geojson"

ORTHO_TILE_URL = (
    "https://tiles.arcgis.com/tiles/2ZRAaoTSJbQ20ceg/"
    "arcgis/rest/services/UiTM_Shah_Alam_Orthomosaic/"
    "MapServer/tile/{z}/{y}/{x}"
)

WEB_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:32647"

BUILDING_COLOUR = "#F39C12"
ROAD_COLOUR = "#8B4513"
ROUTE_COLOUR = "#E74C3C"
START_COLOUR = "#2ECC71"
DESTINATION_COLOUR = "#3498DB"
INTERMEDIATE_COLOUR = "#8E44AD"


# ============================================================
# 4. SESSION STATE
# ============================================================

if "route_result" not in st.session_state:
    st.session_state.route_result = None

if "route_legs" not in st.session_state:
    st.session_state.route_legs = None

if "total_distance" not in st.session_state:
    st.session_state.total_distance = None

if "selected_building_ids" not in st.session_state:
    st.session_state.selected_building_ids = None

if "last_question" not in st.session_state:
    st.session_state.last_question = ""


# ============================================================
# 5. LOAD GIS DATA
# ============================================================

@st.cache_data(show_spinner="Loading GIS layers...")
def load_gis_data():
    if not BUILDING_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {BUILDING_FILE.name}"
        )

    if not ROAD_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {ROAD_FILE.name}"
        )

    buildings = gpd.read_file(BUILDING_FILE)
    roads = gpd.read_file(ROAD_FILE)

    if buildings.crs is None:
        raise ValueError(
            "The building layer has no coordinate system."
        )

    if roads.crs is None:
        raise ValueError(
            "The road layer has no coordinate system."
        )

    if "FID" not in buildings.columns:
        raise ValueError(
            "The building layer requires a field named FID."
        )

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
    buildings, roads = load_gis_data()

except Exception as error:
    st.error(f"Could not load GIS data: {error}")
    st.stop()


buildings_wgs = buildings.to_crs(WEB_CRS)
roads_wgs = roads.to_crs(WEB_CRS)

buildings_m = buildings.to_crs(PROJECTED_CRS)
roads_m = roads.to_crs(PROJECTED_CRS)


# ============================================================
# 6. BUILD ROAD NETWORK
# ============================================================

@st.cache_resource(show_spinner="Preparing road network...")
def build_road_graph(_roads_projected):
    """
    The leading underscore tells Streamlit not to hash the
    GeoDataFrame argument. This fixes the caching error.
    """

    exploded = (
        _roads_projected
        .explode(index_parts=False)
        .reset_index(drop=True)
    )

    exploded = exploded[
        exploded.geometry.notna()
        & ~exploded.geometry.is_empty
        & exploded.geometry.geom_type.eq("LineString")
    ].copy()

    graph = nx.Graph()

    for _, row in exploded.iterrows():
        geometry = row.geometry
        road_id = row.get("FID", -1)

        coordinates = list(geometry.coords)

        for index in range(len(coordinates) - 1):
            point_a = tuple(coordinates[index])
            point_b = tuple(coordinates[index + 1])

            segment = LineString([point_a, point_b])
            distance = float(segment.length)

            if distance <= 0:
                continue

            graph.add_node(point_a)
            graph.add_node(point_b)

            graph.add_edge(
                point_a,
                point_b,
                weight=distance,
                road_id=road_id,
                geometry=segment,
            )

    if graph.number_of_nodes() == 0:
        raise ValueError(
            "No road-network nodes could be created."
        )

    connected_components = list(
        nx.connected_components(graph)
    )

    largest_component = max(
        connected_components,
        key=len,
    )

    main_graph = graph.subgraph(
        largest_component
    ).copy()

    node_list = list(main_graph.nodes)
    node_array = np.asarray(
        node_list,
        dtype=float,
    )

    node_tree = cKDTree(node_array)

    return (
        main_graph,
        node_array,
        node_tree,
        len(connected_components),
    )


try:
    (
        G_MAIN,
        MAIN_NODE_ARRAY,
        MAIN_TREE,
        ROAD_GROUP_COUNT,
    ) = build_road_graph(roads_m)

except Exception as error:
    st.error(
        f"Could not prepare road network: {error}"
    )
    st.stop()


# ============================================================
# 7. ROUTING ENGINE
# ============================================================

def nearest_main_node(point):
    distance, index = MAIN_TREE.query(
        [point.x, point.y]
    )

    node = tuple(
        MAIN_NODE_ARRAY[index]
    )

    return node, float(distance)


def shortest_route_between_buildings(
    origin_fid,
    destination_fid,
):
    origin = buildings_m[
        buildings_m["FID"] == origin_fid
    ]

    destination = buildings_m[
        buildings_m["FID"] == destination_fid
    ]

    if origin.empty:
        raise ValueError(
            f"Building {origin_fid} was not found."
        )

    if destination.empty:
        raise ValueError(
            f"Building {destination_fid} was not found."
        )

    origin_point = (
        origin.geometry
        .representative_point()
        .iloc[0]
    )

    destination_point = (
        destination.geometry
        .representative_point()
        .iloc[0]
    )

    origin_node, origin_snap = nearest_main_node(
        origin_point
    )

    destination_node, destination_snap = (
        nearest_main_node(destination_point)
    )

    path_nodes = nx.shortest_path(
        G_MAIN,
        source=origin_node,
        target=destination_node,
        weight="weight",
    )

    route_segments = []
    total_distance = 0.0

    for index in range(len(path_nodes) - 1):
        node_a = path_nodes[index]
        node_b = path_nodes[index + 1]

        edge = G_MAIN[node_a][node_b]

        route_segments.append(
            edge["geometry"]
        )

        total_distance += float(
            edge["weight"]
        )

    route = gpd.GeoDataFrame(
        geometry=route_segments,
        crs=PROJECTED_CRS,
    )

    return (
        route,
        total_distance,
        origin_snap,
        destination_snap,
    )


def calculate_multi_stop_route(building_ids):
    if len(building_ids) < 2:
        raise ValueError(
            "At least two building IDs are required."
        )

    valid_ids = set(
        buildings_m["FID"]
        .astype(int)
        .tolist()
    )

    missing_ids = [
        building_id
        for building_id in building_ids
        if building_id not in valid_ids
    ]

    if missing_ids:
        raise ValueError(
            f"Building IDs not found: {missing_ids}"
        )

    route_parts = []
    route_legs = []
    total_distance = 0.0

    for index in range(len(building_ids) - 1):
        origin = building_ids[index]
        destination = building_ids[index + 1]

        (
            route,
            leg_distance,
            origin_snap,
            destination_snap,
        ) = shortest_route_between_buildings(
            origin,
            destination,
        )

        route = route.copy()
        route["leg"] = (
            f"{origin} → {destination}"
        )

        route["leg_distance_m"] = (
            leg_distance
        )

        route_parts.append(route)

        route_legs.append(
            {
                "From building": origin,
                "To building": destination,
                "Distance (m)": round(
                    leg_distance,
                    1,
                ),
            }
        )

        total_distance += leg_distance

    combined_route = gpd.GeoDataFrame(
        pd.concat(
            route_parts,
            ignore_index=True,
        ),
        crs=PROJECTED_CRS,
    )

    return (
        combined_route,
        route_legs,
        total_distance,
    )


# ============================================================
# 8. BASIC NATURAL-LANGUAGE PARSER
# ============================================================

def parse_command(question):
    question_lower = question.lower().strip()

    building_ids = [
        int(number)
        for number in re.findall(
            r"\d+",
            question_lower,
        )
    ]

    route_keywords = [
        "route",
        "shortest",
        "path",
        "go",
        "navigate",
        "travel",
        "from",
        "visit",
        "walk",
    ]

    if (
        any(
            keyword in question_lower
            for keyword in route_keywords
        )
        and len(building_ids) >= 2
    ):
        return {
            "action": "route",
            "building_ids": building_ids,
        }

    return {
        "action": "unknown",
        "building_ids": [],
    }


# ============================================================
# 9. TRANSPARENT ORTHOMOSAIC TILE LAYER
# ============================================================

class TransparentWhiteTileLayer(MacroElement):
    def __init__(
        self,
        tile_url,
        white_threshold=245,
        opacity=1.0,
        max_native_zoom=20,
        max_zoom=23,
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

                            var pixels =
                                imageData.data;

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


# ============================================================
# 10. CREATE MAP
# ============================================================

def create_map(
    route_result=None,
    route_legs=None,
    total_distance=None,
    building_ids=None,
):
    (
        minimum_x,
        minimum_y,
        maximum_x,
        maximum_y,
    ) = buildings_wgs.total_bounds

    centre_latitude = (
        minimum_y + maximum_y
    ) / 2

    centre_longitude = (
        minimum_x + maximum_x
    ) / 2

    campus_map = folium.Map(
        location=[
            centre_latitude,
            centre_longitude,
        ],
        zoom_start=16,
        tiles=None,
        max_zoom=23,
        control_scale=True,
        prefer_canvas=True,
    )

    # OpenStreetMap base
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
        show=True,
        max_zoom=23,
    ).add_to(campus_map)

    # Transparent orthomosaic overlay
    orthomosaic_layer = TransparentWhiteTileLayer(
        tile_url=ORTHO_TILE_URL,
        white_threshold=245,
        opacity=1.0,
        max_native_zoom=20,
        max_zoom=23,
    )

    orthomosaic_layer.add_to(campus_map)

    # Road network
    folium.GeoJson(
        roads_wgs,
        name="Road Network",
        style_function=lambda feature: {
            "color": ROAD_COLOUR,
            "weight": 3,
            "opacity": 0.80,
        },
    ).add_to(campus_map)

    # Building footprints
    folium.GeoJson(
        buildings_wgs,
        name="Building Footprints",
        style_function=lambda feature: {
            "color": BUILDING_COLOUR,
            "weight": 1.2,
            "fillColor": BUILDING_COLOUR,
            "fillOpacity": 0.22,
        },
        highlight_function=lambda feature: {
            "color": "#FFD700",
            "weight": 3,
            "fillOpacity": 0.38,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["FID"],
            aliases=["Building ID:"],
            sticky=True,
        ),
    ).add_to(campus_map)

    # Route result
    if (
        route_result is not None
        and building_ids is not None
        and total_distance is not None
    ):
        route_wgs = route_result.to_crs(
            WEB_CRS
        )

        walking_time = (
            total_distance / 80.0
        )

        leg_text = ""

        if route_legs:
            for leg in route_legs:
                leg_text += (
                    f"{leg['From building']} → "
                    f"{leg['To building']}: "
                    f"{leg['Distance (m)']} m<br>"
                )

        route_summary = (
            f"<b>Calculated Route</b><br>"
            f"Stops: {building_ids}<br><br>"
            f"{leg_text}"
            f"<b>Total distance:</b> "
            f"{total_distance:.1f} m<br>"
            f"<b>Walking time:</b> "
            f"{walking_time:.1f} min"
        )

        folium.GeoJson(
            route_wgs,
            name="Calculated Route",
            style_function=lambda feature: {
                "color": ROUTE_COLOUR,
                "weight": 7,
                "opacity": 1.0,
            },
            tooltip=folium.Tooltip(
                route_summary,
                sticky=True,
            ),
        ).add_to(campus_map)

        # Selected stop buildings
        for stop_number, building_id in enumerate(
            building_ids,
            start=1,
        ):
            selected_wgs = buildings_wgs[
                buildings_wgs["FID"]
                == building_id
            ]

            selected_m = buildings_m[
                buildings_m["FID"]
                == building_id
            ]

            if (
                selected_wgs.empty
                or selected_m.empty
            ):
                continue

            marker_point_m = (
                selected_m.geometry
                .representative_point()
                .iloc[0]
            )

            marker_point_wgs = (
                gpd.GeoSeries(
                    [marker_point_m],
                    crs=PROJECTED_CRS,
                )
                .to_crs(WEB_CRS)
                .iloc[0]
            )

            if stop_number == 1:
                colour = START_COLOUR
                label = (
                    f"Start: Building "
                    f"{building_id}"
                )

            elif stop_number == len(
                building_ids
            ):
                colour = DESTINATION_COLOUR
                label = (
                    f"Destination: Building "
                    f"{building_id}"
                )

            else:
                colour = INTERMEDIATE_COLOUR
                label = (
                    f"Stop {stop_number}: "
                    f"Building {building_id}"
                )

            folium.GeoJson(
                selected_wgs,
                name=label,
                style_function=(
                    lambda feature, c=colour: {
                        "color": c,
                        "weight": 3,
                        "fillColor": c,
                        "fillOpacity": 0.75,
                    }
                ),
                tooltip=folium.Tooltip(label),
            ).add_to(campus_map)

            folium.Marker(
                location=[
                    marker_point_wgs.y,
                    marker_point_wgs.x,
                ],
                tooltip=label,
                popup=label,
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        background:{colour};
                        color:white;
                        border-radius:50%;
                        width:28px;
                        height:28px;
                        text-align:center;
                        line-height:28px;
                        font-weight:bold;
                        border:2px solid white;
                        box-shadow:
                            0 1px 5px
                            rgba(0,0,0,0.6);
                    ">
                        {stop_number}
                    </div>
                    """
                ),
            ).add_to(campus_map)

        route_bounds = route_wgs.total_bounds

        campus_map.fit_bounds(
            [
                [
                    route_bounds[1],
                    route_bounds[0],
                ],
                [
                    route_bounds[3],
                    route_bounds[2],
                ],
            ],
            padding=(45, 45),
        )

    else:
        campus_map.fit_bounds(
            [
                [minimum_y, minimum_x],
                [maximum_y, maximum_x],
            ],
            padding=(20, 20),
        )

    Fullscreen(
        position="topright",
        title="Full screen",
        title_cancel="Exit full screen",
        force_separate_button=True,
    ).add_to(campus_map)

    layer_control = folium.LayerControl(
        collapsed=False,
        position="topleft",
    )

    layer_control.add_to(campus_map)

    # Register custom orthomosaic in layer control
    campus_map.get_root().script.add_child(
        folium.Element(
            f"""
            setTimeout(function() {{
                try {{
                    {layer_control.get_name()}.addOverlay(
                        {orthomosaic_layer.get_name()}_layer,
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

    return campus_map


# ============================================================
# 11. STREAMLIT HEADER
# ============================================================

st.title("🗺️ AI GIS Dashboard")

st.caption(
    "Natural-language campus routing using UAV orthomosaic, "
    "building footprints and road-network analysis."
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Buildings",
        f"{len(buildings_wgs):,}",
    )

with metric2:
    st.metric(
        "Road features",
        f"{len(roads_wgs):,}",
    )

with metric3:
    st.metric(
        "Road-network groups",
        ROAD_GROUP_COUNT,
    )


# ============================================================
# 12. COMMAND FORM
# ============================================================

st.subheader("Ask the GIS")

with st.form(
    key="gis_command_form",
    clear_on_submit=False,
):
    question = st.text_input(
        "Enter a route command",
        value=st.session_state.last_question,
        placeholder=(
            "Example: route from building 10 "
            "to building 20 to building 35"
        ),
    )

    submitted = st.form_submit_button(
        "Run GIS command",
        type="primary",
    )


if submitted:
    st.session_state.last_question = question

    if not question.strip():
        st.warning(
            "Please enter a route command."
        )

    else:
        parsed = parse_command(question)

        if parsed["action"] == "route":
            building_ids = parsed[
                "building_ids"
            ]

            try:
                with st.spinner(
                    "Calculating shortest route..."
                ):
                    (
                        route_result,
                        route_legs,
                        total_distance,
                    ) = calculate_multi_stop_route(
                        building_ids
                    )

                st.session_state.route_result = (
                    route_result
                )

                st.session_state.route_legs = (
                    route_legs
                )

                st.session_state.total_distance = (
                    total_distance
                )

                st.session_state.selected_building_ids = (
                    building_ids
                )

                st.success(
                    "Route calculated successfully."
                )

            except Exception as error:
                st.error(
                    f"Unable to calculate route: "
                    f"{error}"
                )

        else:
            st.warning(
                "Command not recognised. Try: "
                "'route from building 10 "
                "to building 20'."
            )


# ============================================================
# 13. CLEAR ROUTE
# ============================================================

if st.session_state.route_result is not None:
    if st.button("Clear route"):
        st.session_state.route_result = None
        st.session_state.route_legs = None
        st.session_state.total_distance = None
        st.session_state.selected_building_ids = None
        st.session_state.last_question = ""
        st.rerun()


# ============================================================
# 14. ROUTE RESULTS
# ============================================================

if (
    st.session_state.route_result is not None
    and st.session_state.total_distance
    is not None
):
    total_distance = (
        st.session_state.total_distance
    )

    walking_time = total_distance / 80.0

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "Stops",
            len(
                st.session_state
                .selected_building_ids
            ),
        )

    with result2:
        st.metric(
            "Total distance",
            f"{total_distance:.1f} m",
        )

    with result3:
        st.metric(
            "Estimated walking time",
            f"{walking_time:.1f} min",
        )

    route_table = pd.DataFrame(
        st.session_state.route_legs
    )

    st.dataframe(
        route_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 15. DISPLAY MAP
# ============================================================

campus_map = create_map(
    route_result=st.session_state.route_result,
    route_legs=st.session_state.route_legs,
    total_distance=st.session_state.total_distance,
    building_ids=(
        st.session_state
        .selected_building_ids
    ),
)

st_folium(
    campus_map,
    width=None,
    height=720,
    returned_objects=[],
    use_container_width=True,
)
