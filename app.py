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
# 1. STREAMLIT SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI GIS Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. DASHBOARD CSS
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
    }

    iframe {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. DATA PATHS AND SETTINGS
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
STOP_COLOUR = "#8E44AD"


# ============================================================
# 4. LOAD DATA
# ============================================================

@st.cache_data(show_spinner="Loading GIS data...")
def load_gis_data():
    if not BUILDING_FILE.exists():
        raise FileNotFoundError(BUILDING_FILE.name)

    if not ROAD_FILE.exists():
        raise FileNotFoundError(ROAD_FILE.name)

    buildings = gpd.read_file(BUILDING_FILE)
    roads = gpd.read_file(ROAD_FILE)

    if buildings.crs is None or roads.crs is None:
        raise ValueError("One or more GIS layers have no CRS.")

    if "FID" not in buildings.columns:
        raise ValueError("Building layer requires an FID field.")

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
# 5. BUILD ROAD NETWORK
# ============================================================

@st.cache_resource(show_spinner="Preparing road network...")
def build_road_graph(roads_projected):
    exploded = (
        roads_projected
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
        raise ValueError("Road graph contains no nodes.")

    largest_component = max(
        nx.connected_components(graph),
        key=len,
    )

    main_graph = graph.subgraph(largest_component).copy()

    node_list = list(main_graph.nodes)
    node_array = np.array(node_list, dtype=float)
    node_tree = cKDTree(node_array)

    return main_graph, node_array, node_tree


try:
    G_MAIN, MAIN_NODE_ARRAY, MAIN_TREE = build_road_graph(roads_m)

except Exception as error:
    st.error(f"Could not prepare road network: {error}")
    st.stop()


# ============================================================
# 6. ROUTING FUNCTIONS
# ============================================================

def nearest_main_node(point):
    distance, index = MAIN_TREE.query([point.x, point.y])

    node = tuple(MAIN_NODE_ARRAY[index])

    return node, float(distance)


def shortest_route_between_buildings(origin_fid, destination_fid):
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

    origin_point = origin.geometry.centroid.iloc[0]
    destination_point = destination.geometry.centroid.iloc[0]

    origin_node, origin_snap = nearest_main_node(origin_point)
    destination_node, destination_snap = nearest_main_node(
        destination_point
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

        route_segments.append(edge["geometry"])
        total_distance += float(edge["weight"])

    route = gpd.GeoDataFrame(
        geometry=route_segments,
        crs=PROJECTED_CRS,
    )

    return route, total_distance, origin_snap, destination_snap


def calculate_multi_stop_route(building_ids):
    if len(building_ids) < 2:
        raise ValueError(
            "At least two building IDs are required."
        )

    valid_ids = set(
        buildings_m["FID"].astype(int).tolist()
    )

    missing = [
        building_id
        for building_id in building_ids
        if building_id not in valid_ids
    ]

    if missing:
        raise ValueError(
            f"Building IDs not found: {missing}"
        )

    route_parts = []
    route_legs = []
    total_distance = 0.0

    for index in range(len(building_ids) - 1):
        origin = building_ids[index]
        destination = building_ids[index + 1]

        route, leg_distance, origin_snap, destination_snap = (
            shortest_route_between_buildings(
                origin,
                destination,
            )
        )

        route = route.copy()
        route["leg"] = f"{origin} → {destination}"
        route["leg_distance_m"] = leg_distance

        route_parts.append(route)

        route_legs.append(
            {
                "origin": origin,
                "destination": destination,
                "distance_m": leg_distance,
            }
        )

        total_distance += leg_distance

    combined_route = gpd.GeoDataFrame(
        pd.concat(route_parts, ignore_index=True),
        crs=PROJECTED_CRS,
    )

    return combined_route, route_legs, total_distance


# ============================================================
# 7. SIMPLE NATURAL-LANGUAGE PARSER
# ============================================================

def parse_command(question):
    question_lower = question.lower().strip()

    building_ids = [
        int(number)
        for number in re.findall(r"\d+", question_lower)
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
# 8. TRANSPARENT ORTHOMOSAIC
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

            var {{ this.get_name() }} = L.GridLayer.extend({

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

                                if (nearWhite && similarValues) {
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
                new {{ this.get_name() }}({

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
# 9. CREATE MAP
# ============================================================

def create_map(
    route_result=None,
    route_legs=None,
    total_distance=None,
    building_ids=None,
):
    minimum_x, minimum_y, maximum_x, maximum_y = (
        buildings_wgs.total_bounds
    )

    centre_latitude = (minimum_y + maximum_y) / 2
    centre_longitude = (minimum_x + maximum_x) / 2

    campus_map = folium.Map(
        location=[centre_latitude, centre_longitude],
        zoom_start=16,
        tiles=None,
        max_zoom=23,
        control_scale=True,
        prefer_canvas=True,
    )

    openstreetmap = folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
        show=True,
        max_zoom=23,
    )

    openstreetmap.add_to(campus_map)

    orthomosaic = TransparentWhiteTileLayer(
        tile_url=ORTHO_TILE_URL,
        white_threshold=245,
        opacity=1.0,
        max_native_zoom=20,
        max_zoom=23,
    )

    orthomosaic.add_to(campus_map)

    folium.GeoJson(
        roads_wgs,
        name="Road Network",
        style_function=lambda feature: {
            "color": ROAD_COLOUR,
            "weight": 3,
            "opacity": 0.80,
        },
    ).add_to(campus_map)

    folium.GeoJson(
        buildings_wgs,
        name="Building Footprints",
        style_function=lambda feature: {
            "color": BUILDING_COLOUR,
            "weight": 1.2,
            "fillColor": BUILDING_COLOUR,
            "fillOpacity": 0.22,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["FID"],
            aliases=["Building ID:"],
            sticky=True,
        ),
    ).add_to(campus_map)

    if route_result is not None:
        route_wgs = route_result.to_crs(WEB_CRS)

        walking_time = total_distance / 80.0

        route_summary = (
            f"<b>Calculated Route</b><br>"
            f"Stops: {building_ids}<br>"
            f"Total distance: {total_distance:.2f} m<br>"
            f"Estimated walking time: "
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

        for stop_number, building_id in enumerate(
            building_ids,
            start=1,
        ):
            selected = buildings_wgs[
                buildings_wgs["FID"] == building_id
            ]

            if selected.empty:
                continue

            centroid = selected.geometry.centroid.iloc[0]

            if stop_number == 1:
                colour = START_COLOUR
                label = f"Start: Building {building_id}"

            elif stop_number == len(building_ids):
                colour = DESTINATION_COLOUR
                label = (
                    f"Destination: Building {building_id}"
                )

            else:
                colour = STOP_COLOUR
                label = (
                    f"Stop {stop_number}: "
                    f"Building {building_id}"
                )

            folium.GeoJson(
                selected,
                name=label,
                style_function=lambda feature, c=colour: {
                    "color": c,
                    "weight": 3,
                    "fillColor": c,
                    "fillOpacity": 0.75,
                },
                tooltip=folium.Tooltip(label),
            ).add_to(campus_map)

            folium.Marker(
                location=[centroid.y, centroid.x],
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
                        box-shadow:0 1px 5px rgba(0,0,0,0.6);
                    ">
                        {stop_number}
                    </div>
                    """
                ),
            ).add_to(campus_map)

        route_bounds = route_wgs.total_bounds

        campus_map.fit_bounds(
            [
                [route_bounds[1], route_bounds[0]],
                [route_bounds[3], route_bounds[2]],
            ],
            padding=(40, 40),
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
    ).add_to(campus_map)

    folium.LayerControl(
        collapsed=False,
        position="topleft",
    ).add_to(campus_map)

    return campus_map


# ============================================================
# 10. STREAMLIT INTERFACE
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
        "Current capability",
        "Route planner",
    )


# ============================================================
# 11. COMMAND BOX
# ============================================================

st.subheader("Ask the GIS")

question = st.text_input(
    "Enter a command",
    placeholder=(
        "Example: route from building 10 "
        "to building 20 to building 35"
    ),
)

run_button = st.button(
    "Run GIS command",
    type="primary",
)

route_result = None
route_legs = None
total_distance = None
selected_building_ids = None


if run_button:
    if not question.strip():
        st.warning("Please enter a GIS command.")

    else:
        parsed = parse_command(question)

        if parsed["action"] == "route":
            selected_building_ids = parsed[
                "building_ids"
            ]

            try:
                with st.spinner(
                    "Calculating the shortest route..."
                ):
                    (
                        route_result,
                        route_legs,
                        total_distance,
                    ) = calculate_multi_stop_route(
                        selected_building_ids
                    )

                walking_time = total_distance / 80.0

                st.success(
                    "Route calculated successfully."
                )

                result1, result2, result3 = st.columns(3)

                with result1:
                    st.metric(
                        "Stops",
                        len(selected_building_ids),
                    )

                with result2:
                    st.metric(
                        "Total distance",
                        f"{total_distance:.1f} m",
                    )

                with result3:
                    st.metric(
                        "Walking time",
                        f"{walking_time:.1f} min",
                    )

                route_table = pd.DataFrame(route_legs)

                route_table.columns = [
                    "From building",
                    "To building",
                    "Distance (m)",
                ]

                route_table["Distance (m)"] = (
                    route_table["Distance (m)"]
                    .round(1)
                )

                st.dataframe(
                    route_table,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as error:
                st.error(
                    f"Unable to calculate route: {error}"
                )

        else:
            st.warning(
                "Command not recognised. Try: "
                "'route from building 10 to building 20'."
            )


# ============================================================
# 12. DISPLAY MAP
# ============================================================

campus_map = create_map(
    route_result=route_result,
    route_legs=route_legs,
    total_distance=total_distance,
    building_ids=selected_building_ids,
)

st_folium(
    campus_map,
    width=None,
    height=720,
    returned_objects=[],
    use_container_width=True,
)
