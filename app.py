from pathlib import Path
import math
import re

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from ai_engine import interpret_gis_command

from branca.element import MacroElement
from folium.plugins import Fullscreen, PolyLineTextPath
from jinja2 import Template
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from streamlit_folium import st_folium
from building_manager import (
    ensure_name_column,
    building_display_label,
    update_building_name_locally,
    save_building_names_to_github,
)


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

# Unique stop-marker colours for up to 10 stops.
STOP_COLOURS = [
    "#1976D2",  # 1 blue
    "#EF6C00",  # 2 orange
    "#2E7D32",  # 3 green
    "#7B1FA2",  # 4 purple
    "#D32F2F",  # 5 red
    "#795548",  # 6 brown
    "#D81B60",  # 7 pink
    "#616161",  # 8 grey
    "#C0A000",  # 9 gold
    "#0097A7",  # 10 cyan
]

TRAVEL_SPEEDS_KMH = {
    "Walking": 4.8,
    "E-bike": 18.0,
    "Motorcycle": 30.0,
    "Car driving": 45.0,
}


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

# Ensure every building has a clean NAME field.
buildings = ensure_name_column(buildings)

buildings_wgs = ensure_name_column(
    buildings.to_crs(WEB_CRS)
)
roads_wgs = roads.to_crs(WEB_CRS)

buildings_m = ensure_name_column(
    buildings.to_crs(PROJECTED_CRS)
)
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


def get_building_name(building_id):
    row = buildings[buildings["FID"] == int(building_id)]
    if row.empty:
        return f"Building {building_id}"

    name = str(row.iloc[0].get("NAME", "")).strip()
    return name if name else f"Building {building_id}"


def format_travel_time(distance_m, speed_kmh):
    """Return a readable estimated travel time."""
    total_seconds = max(0, round((float(distance_m) / 1000.0) / speed_kmh * 3600.0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours} hr {minutes} min"
    if minutes:
        return f"{minutes} min {seconds} sec"
    return f"{seconds} sec"


def travel_time_estimates(distance_m):
    return {
        mode: format_travel_time(distance_m, speed)
        for mode, speed in TRAVEL_SPEEDS_KMH.items()
    }


def offset_marker_positions(building_ids, projected_buildings, offset_m=4.0):
    """
    Return one WGS84 marker point for every stop. Repeated visits to the
    same building are offset around the representative point so that, for
    example, Stop 1 and Stop 6 remain visible instead of overlapping.
    """
    counts = {}
    for fid in building_ids:
        counts[int(fid)] = counts.get(int(fid), 0) + 1

    used = {}
    points = []

    for fid in building_ids:
        fid = int(fid)
        selected = projected_buildings[projected_buildings["FID"] == fid]
        if selected.empty:
            points.append(None)
            continue

        base = selected.geometry.representative_point().iloc[0]
        duplicate_total = counts[fid]
        duplicate_index = used.get(fid, 0)
        used[fid] = duplicate_index + 1

        if duplicate_total > 1:
            angle = (2 * math.pi * duplicate_index / duplicate_total) - (math.pi / 2)
            x = base.x + offset_m * math.cos(angle)
            y = base.y + offset_m * math.sin(angle)
        else:
            x, y = base.x, base.y

        wgs_point = (
            gpd.GeoSeries([LineString([(x, y), (x, y)]).centroid], crs=PROJECTED_CRS)
            .to_crs(WEB_CRS)
            .iloc[0]
        )
        points.append(wgs_point)

    return points


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
            fields=["FID", "NAME"],
            aliases=["Building ID:", "Building Name:"],
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
                    f"{get_building_name(leg['From building'])} → "
                    f"{get_building_name(leg['To building'])}: "
                    f"{leg['Distance (m)']} m<br>"
                )

        times = travel_time_estimates(total_distance)

        route_summary = (
            f"<b>Calculated Route</b><br>"
            f"Stops: {building_ids}<br><br>"
            f"{leg_text}"
            f"<b>Total distance:</b> {total_distance:.1f} m<br>"
            f"<b>Walking:</b> {times['Walking']}<br>"
            f"<b>E-bike:</b> {times['E-bike']}<br>"
            f"<b>Motorcycle:</b> {times['Motorcycle']}<br>"
            f"<b>Car driving:</b> {times['Car driving']}"
        )

        # Draw route with repeated directional arrows.
        route_group = folium.FeatureGroup(
            name="Calculated Route (with arrows)",
            show=True,
        )
        route_group.add_to(campus_map)

        for geometry in route_wgs.geometry:
            if geometry is None or geometry.is_empty:
                continue

            line_geometries = (
                list(geometry.geoms)
                if geometry.geom_type == "MultiLineString"
                else [geometry]
            )

            for line_geometry in line_geometries:
                coordinates = [
                    [latitude, longitude]
                    for longitude, latitude in line_geometry.coords
                ]

                # White casing improves visibility over the orthomosaic.
                folium.PolyLine(
                    coordinates,
                    color="white",
                    weight=10,
                    opacity=0.90,
                ).add_to(route_group)

                route_line = folium.PolyLine(
                    coordinates,
                    color=ROUTE_COLOUR,
                    weight=6,
                    opacity=1.0,
                    tooltip=folium.Tooltip(route_summary, sticky=True),
                )
                route_line.add_to(route_group)

                PolyLineTextPath(
                    route_line,
                    "➤",
                    repeat=True,
                    offset=7,
                    attributes={
                        "fill": "white",
                        "font-weight": "bold",
                        "font-size": "17",
                    },
                ).add_to(campus_map)

        # Unique stop colours and offset repeated visits so markers do not overlap.
        marker_points = offset_marker_positions(
            building_ids,
            buildings_m,
            offset_m=5.0,
        )

        for stop_number, (building_id, marker_point_wgs) in enumerate(
            zip(building_ids, marker_points),
            start=1,
        ):
            selected_wgs = buildings_wgs[
                buildings_wgs["FID"] == building_id
            ]

            if selected_wgs.empty or marker_point_wgs is None:
                continue

            colour = STOP_COLOURS[(stop_number - 1) % len(STOP_COLOURS)]

            if stop_number == 1:
                label = f"Start: {get_building_name(building_id)}"
            elif stop_number == len(building_ids):
                label = f"Destination: {get_building_name(building_id)}"
            else:
                label = f"Stop {stop_number}: {get_building_name(building_id)}"

            folium.GeoJson(
                selected_wgs,
                name=label,
                style_function=(
                    lambda feature, c=colour: {
                        "color": c,
                        "weight": 3,
                        "fillColor": c,
                        "fillOpacity": 0.65,
                    }
                ),
                tooltip=folium.Tooltip(label),
            ).add_to(campus_map)

            folium.Marker(
                location=[marker_point_wgs.y, marker_point_wgs.x],
                tooltip=label,
                popup=label,
                z_index_offset=1000 + stop_number,
                icon=folium.DivIcon(
                    icon_size=(34, 34),
                    icon_anchor=(17, 17),
                    html=f"""
                    <div style="
                        background:{colour};
                        color:white;
                        border-radius:50%;
                        width:32px;
                        height:32px;
                        text-align:center;
                        line-height:32px;
                        font-weight:800;
                        font-size:15px;
                        border:3px solid white;
                        box-shadow:0 2px 7px rgba(0,0,0,0.75);
                    ">
                        {stop_number}
                    </div>
                    """,
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
# 12. BUILDING NAME EDITOR
# ============================================================

with st.expander("🏢 Admin Building Name Editor", expanded=False):
    admin_password = st.text_input(
        "Admin password",
        type="password",
        key="admin_password_input",
    )

    if admin_password:
        expected_password = str(
            st.secrets.get("ADMIN_PASSWORD", "")
        )

        if admin_password != expected_password:
            st.error("Incorrect admin password.")
        else:
            st.success("Admin access enabled.")

            editor_buildings = buildings.sort_values("FID").copy()
            editor_options = {
                building_display_label(row): int(row["FID"])
                for _, row in editor_buildings.iterrows()
            }

            selected_editor_label = st.selectbox(
                "Select building",
                options=list(editor_options.keys()),
                key="building_editor_select",
            )

            selected_editor_fid = editor_options[
                selected_editor_label
            ]

            selected_record = buildings[
                buildings["FID"] == selected_editor_fid
            ].iloc[0]

            current_name = str(
                selected_record.get("NAME", "")
            ).strip()

            st.write(f"**Building FID:** {selected_editor_fid}")
            st.write(
                "**Current name:** "
                + (current_name if current_name else "Unnamed")
            )

            # ----------------------------------------------------
            # Selected-building preview map
            # ----------------------------------------------------
            st.markdown("#### Selected building preview")
            st.caption(
                "Change the FID above and this map will zoom to the "
                "selected building so you can identify it before naming it."
            )

            preview_selected_wgs = buildings_wgs[
                buildings_wgs["FID"] == selected_editor_fid
            ].copy()

            preview_selected_m = buildings_m[
                buildings_m["FID"] == selected_editor_fid
            ].copy()

            if not preview_selected_wgs.empty:
                preview_point_m = (
                    preview_selected_m.geometry
                    .representative_point()
                    .iloc[0]
                )

                preview_point_wgs = (
                    gpd.GeoSeries(
                        [preview_point_m],
                        crs=PROJECTED_CRS,
                    )
                    .to_crs(WEB_CRS)
                    .iloc[0]
                )

                preview_map = folium.Map(
                    location=[
                        preview_point_wgs.y,
                        preview_point_wgs.x,
                    ],
                    zoom_start=20,
                    tiles=None,
                    max_zoom=23,
                    control_scale=True,
                )

                # OpenStreetMap base
                folium.TileLayer(
                    tiles="OpenStreetMap",
                    name="OpenStreetMap",
                    overlay=False,
                    control=True,
                    show=True,
                    max_zoom=23,
                ).add_to(preview_map)

                # Transparent UAV orthomosaic overlay
                preview_ortho = TransparentWhiteTileLayer(
                    tile_url=ORTHO_TILE_URL,
                    white_threshold=245,
                    opacity=1.0,
                    max_native_zoom=20,
                    max_zoom=23,
                )
                preview_ortho.add_to(preview_map)

                # Nearby roads for orientation
                folium.GeoJson(
                    roads_wgs,
                    name="Road Network",
                    style_function=lambda feature: {
                        "color": ROAD_COLOUR,
                        "weight": 2.5,
                        "opacity": 0.75,
                    },
                ).add_to(preview_map)

                # All buildings remain orange
                folium.GeoJson(
                    buildings_wgs,
                    name="Building Footprints",
                    style_function=lambda feature: {
                        "color": BUILDING_COLOUR,
                        "weight": 1.0,
                        "fillColor": BUILDING_COLOUR,
                        "fillOpacity": 0.18,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=["FID", "NAME"],
                        aliases=["Building ID:", "Building Name:"],
                        sticky=True,
                    ),
                ).add_to(preview_map)

                # Selected building is highlighted in green
                folium.GeoJson(
                    preview_selected_wgs,
                    name=f"Selected Building {selected_editor_fid}",
                    style_function=lambda feature: {
                        "color": START_COLOUR,
                        "weight": 4,
                        "fillColor": START_COLOUR,
                        "fillOpacity": 0.72,
                    },
                    tooltip=folium.Tooltip(
                        f"Selected Building FID: {selected_editor_fid}<br>"
                        f"Current name: "
                        f"{current_name if current_name else 'Unnamed'}",
                        sticky=True,
                    ),
                ).add_to(preview_map)

                folium.Marker(
                    location=[
                        preview_point_wgs.y,
                        preview_point_wgs.x,
                    ],
                    tooltip=f"Building FID {selected_editor_fid}",
                    popup=(
                        f"<b>Building FID:</b> {selected_editor_fid}<br>"
                        f"<b>Current name:</b> "
                        f"{current_name if current_name else 'Unnamed'}"
                    ),
                ).add_to(preview_map)

                folium.LayerControl(
                    collapsed=True,
                    position="topright",
                ).add_to(preview_map)

                st_folium(
                    preview_map,
                    width=None,
                    height=430,
                    returned_objects=[],
                    use_container_width=True,
                    key=f"building_preview_map_{selected_editor_fid}",
                )
            else:
                st.warning(
                    f"Building FID {selected_editor_fid} could not be displayed."
                )

            new_name = st.text_input(
                "New building name",
                value=current_name,
                key=f"new_building_name_{selected_editor_fid}",
            )

            if st.button(
                "Save building name",
                type="primary",
                key="save_building_name_button",
            ):
                try:
                    updated_buildings = update_building_name_locally(
                        buildings=buildings,
                        building_fid=selected_editor_fid,
                        new_name=new_name,
                    )

                    save_building_names_to_github(
                        buildings=updated_buildings,
                        streamlit_secrets=st.secrets,
                        building_fid=selected_editor_fid,
                        new_name=new_name,
                    )

                    st.success(
                        f"Saved '{new_name}' for Building "
                        f"{selected_editor_fid}."
                    )
                    st.info(
                        "GitHub has been updated. The app will "
                        "redeploy automatically."
                    )
                    st.cache_data.clear()

                except Exception as error:
                    st.error(
                        f"Unable to save building name: {error}"
                    )


# ============================================================
# 13. SEARCHABLE BUILDING ROUTE PLANNER
# ============================================================

st.subheader("Building Route Planner")

route_buildings = buildings.copy()
route_buildings["display_label"] = route_buildings.apply(
    building_display_label,
    axis=1,
)
route_buildings = route_buildings.sort_values(
    ["NAME", "FID"],
    na_position="last",
)

building_label_to_fid = dict(
    zip(
        route_buildings["display_label"],
        route_buildings["FID"].astype(int),
    )
)
building_labels = list(building_label_to_fid.keys())

route_col1, route_col2 = st.columns(2)

with route_col1:
    start_label = st.selectbox(
        "Start building",
        options=building_labels,
        key="start_building_select",
    )

with route_col2:
    destination_label = st.selectbox(
        "Destination building",
        options=building_labels,
        index=1 if len(building_labels) > 1 else 0,
        key="destination_building_select",
    )

selected_stop_labels = st.multiselect(
    "Optional intermediate stops",
    options=building_labels,
    key="intermediate_stop_select",
)

if st.button(
    "Calculate route from selected buildings",
    type="primary",
    key="calculate_named_route_button",
):
    try:
        route_ids = [building_label_to_fid[start_label]]
        route_ids.extend(
            building_label_to_fid[label]
            for label in selected_stop_labels
        )
        route_ids.append(
            building_label_to_fid[destination_label]
        )

        if route_ids[0] == route_ids[-1] and len(route_ids) == 2:
            raise ValueError(
                "Start and destination must be different."
            )

        with st.spinner("Calculating shortest route..."):
            route_result, route_legs, total_distance = (
                calculate_multi_stop_route(route_ids)
            )

        st.session_state.route_result = route_result
        st.session_state.route_legs = route_legs
        st.session_state.total_distance = total_distance
        st.session_state.selected_building_ids = route_ids
        st.success("Route calculated successfully.")

    except Exception as error:
        st.error(f"Unable to calculate route: {error}")


# ============================================================
# 14. AI GIS COMMAND FORM
# ============================================================

st.subheader("AI GIS Assistant")

with st.form(
    key="gis_command_form",
    clear_on_submit=False,
):
    question = st.text_input(
        "Enter a route request",
        value=st.session_state.last_question,
        placeholder=(
            "Example: Start at Building 10, visit 20, 35, 50 and return to 10"
        ),
    )

    submitted = st.form_submit_button(
        "Ask GIS",
        type="primary",
    )


if submitted:
    st.session_state.last_question = question

    if not question.strip():
        st.warning("Please enter a route request.")

    else:
        parsed = None
        interpreter_used = "Rule-based fallback"

        try:
            with st.spinner("Gemini is interpreting your request..."):
                parsed = interpret_gis_command(
                    question=question,
                    api_key=st.secrets["GEMINI_API_KEY"],
                )
            interpreter_used = parsed.get("model_used", "Gemini")
            st.success(f"Last language interpreter used: {interpreter_used}")
            if parsed.get("reply"):
                st.info(parsed["reply"])

        except Exception as gemini_error:
            parsed = parse_command(question)
            st.warning(
                "Gemini was unavailable, so the app used the rule-based "
                f"fallback. Details: {gemini_error}"
            )

        if parsed and parsed.get("action") == "route":
            building_ids = [int(value) for value in parsed.get("building_ids", [])]

            if len(building_ids) > 10:
                st.warning(
                    "The map supports unique stop colours for the first 10 stops. "
                    "Please use 10 stops or fewer for the clearest display."
                )

            try:
                with st.spinner("Running GIS shortest-path analysis..."):
                    route_result, route_legs, total_distance = (
                        calculate_multi_stop_route(building_ids)
                    )

                st.session_state.route_result = route_result
                st.session_state.route_legs = route_legs
                st.session_state.total_distance = total_distance
                st.session_state.selected_building_ids = building_ids

                st.success("Route calculated successfully.")

            except Exception as error:
                st.error(f"Unable to calculate route: {error}")

        else:
            st.warning(
                "The request was not recognised as a route. Include at least "
                "two building FIDs, for example: route from Building 10 to Building 20."
            )


# ============================================================
# 15. CLEAR ROUTE
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
# 16. DISPLAY MAP
# ============================================================

campus_map = create_map(
    route_result=st.session_state.route_result,
    route_legs=st.session_state.route_legs,
    total_distance=st.session_state.total_distance,
    building_ids=st.session_state.selected_building_ids,
)

st_folium(
    campus_map,
    width=None,
    height=720,
    returned_objects=[],
    use_container_width=True,
    key="main_campus_map",
)


# ============================================================
# 17. ROUTE RESULTS
# ============================================================

if (
    st.session_state.route_result is not None
    and st.session_state.total_distance is not None
):
    total_distance = st.session_state.total_distance
    times = travel_time_estimates(total_distance)

    st.markdown("### Route summary")

    summary1, summary2 = st.columns(2)
    with summary1:
        st.metric(
            "Stops",
            len(st.session_state.selected_building_ids),
        )
    with summary2:
        st.metric(
            "Total road distance",
            f"{total_distance:.1f} m",
        )

    time1, time2, time3, time4 = st.columns(4)
    with time1:
        st.metric("🚶 Walking", times["Walking"])
        st.caption("Assumed average speed: 4.8 km/h")
    with time2:
        st.metric("🚲 E-bike", times["E-bike"])
        st.caption("Assumed average speed: 18 km/h")
    with time3:
        st.metric("🏍️ Motorcycle", times["Motorcycle"])
        st.caption("Assumed average speed: 30 km/h")
    with time4:
        st.metric("🚗 Car driving", times["Car driving"])
        st.caption("Assumed average speed: 45 km/h")

    st.caption(
        "Travel times are simple estimates based on the same road-network "
        "distance and assumed average speeds. They do not yet account for "
        "traffic, road restrictions, junction delays, parking or vehicle access."
    )

    route_table = pd.DataFrame(st.session_state.route_legs)
    st.dataframe(
        route_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Stop colour legend")
    legend_columns = st.columns(min(10, len(st.session_state.selected_building_ids)))
    for index, building_id in enumerate(st.session_state.selected_building_ids):
        colour = STOP_COLOURS[index % len(STOP_COLOURS)]
        with legend_columns[index % len(legend_columns)]:
            st.markdown(
                f"""
                <div style="text-align:center; margin-bottom:8px;">
                    <span style="
                        display:inline-block;
                        width:30px;
                        height:30px;
                        line-height:30px;
                        border-radius:50%;
                        background:{colour};
                        color:white;
                        font-weight:800;
                        border:2px solid white;
                        box-shadow:0 1px 4px rgba(0,0,0,.35);
                    ">{index + 1}</span><br>
                    <small>{get_building_name(building_id)}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

