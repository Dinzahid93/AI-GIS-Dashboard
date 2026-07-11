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
from folium.plugins import Fullscreen
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
ROAD_COLOUR = "#FFC0CB"
ROUTE_COLOUR = "#E74C3C"
MULTI_ROUTE_COLOURS = [
    "#E74C3C", "#2563EB", "#16A34A", "#9333EA",
    "#EA580C", "#0891B2", "#DB2777", "#4F46E5",
]
STOP_COLOUR = "#0000FF"  # All route stops use yellow
START_COLOUR = STOP_COLOUR
DESTINATION_COLOUR = STOP_COLOUR
INTERMEDIATE_COLOUR = STOP_COLOUR

TRAVEL_SPEEDS_KMH = {
    "Walking": 4.8,
    "E-bike": 18.0,
    "Motorcycle": 30.0,
    "Car driving": 45.0,
}


SERVICE_AREA_COLOURS = {
    "Walking": "#16A34A",
    "E-bike": "#2563EB",
    "Motorcycle": "#EA580C",
    "Car driving": "#9333EA",
}

SERVICE_AREA_ICONS = {
    "Walking": "🚶",
    "E-bike": "🚲",
    "Motorcycle": "🏍️",
    "Car driving": "🚗",
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

if "last_ai_reply" not in st.session_state:
    st.session_state.last_ai_reply = ""

if "last_interpreter" not in st.session_state:
    st.session_state.last_interpreter = ""

if "multi_route_results" not in st.session_state:
    st.session_state.multi_route_results = None

if "multi_route_table" not in st.session_state:
    st.session_state.multi_route_table = None

if "multi_route_destination" not in st.session_state:
    st.session_state.multi_route_destination = None

if "independent_route_mode" not in st.session_state:
    st.session_state.independent_route_mode = None


if "service_area_results" not in st.session_state:
    st.session_state.service_area_results = None

if "service_area_table" not in st.session_state:
    st.session_state.service_area_table = None

if "service_area_origin" not in st.session_state:
    st.session_state.service_area_origin = None

if "service_area_minutes" not in st.session_state:
    st.session_state.service_area_minutes = None


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

        edge_geometry = edge["geometry"]
        edge_coordinates = list(edge_geometry.coords)

        # Ensure every route segment follows the actual travel direction.
        if tuple(edge_coordinates[0]) != tuple(node_a):
            edge_coordinates.reverse()

        route_segments.append(
            LineString(edge_coordinates)
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


def calculate_routes_from_origins_to_destination(
    origin_ids,
    destination_id,
):
    """Calculate one independent shortest path per origin to one destination."""

    origin_ids = list(dict.fromkeys(int(fid) for fid in origin_ids))
    destination_id = int(destination_id)

    if not origin_ids:
        raise ValueError("Select at least one origin building.")

    valid_ids = set(buildings_m["FID"].astype(int).tolist())
    requested_ids = origin_ids + [destination_id]
    missing_ids = [fid for fid in requested_ids if fid not in valid_ids]

    if missing_ids:
        raise ValueError(f"Building IDs not found: {missing_ids}")

    if destination_id in origin_ids:
        origin_ids = [fid for fid in origin_ids if fid != destination_id]

    if not origin_ids:
        raise ValueError("Origins must be different from the destination.")

    route_results = []
    summary_rows = []

    for route_number, origin_id in enumerate(origin_ids, start=1):
        route, distance_m, origin_snap, destination_snap = (
            shortest_route_between_buildings(origin_id, destination_id)
        )

        colour = MULTI_ROUTE_COLOURS[(route_number - 1) % len(MULTI_ROUTE_COLOURS)]
        route = route.copy()
        route["origin_id"] = origin_id
        route["destination_id"] = destination_id
        route["route_label"] = f"{origin_id} → {destination_id}"
        route["route_colour"] = colour
        route["distance_m"] = float(distance_m)

        route_results.append({
            "origin_id": origin_id,
            "destination_id": destination_id,
            "route": route,
            "distance_m": float(distance_m),
            "colour": colour,
        })

        row = {
            "From building": origin_id,
            "From name": get_building_name(origin_id),
            "To building": destination_id,
            "To name": get_building_name(destination_id),
            "Distance (m)": round(float(distance_m), 1),
        }

        distance_km = float(distance_m) / 1000.0
        for mode_name, speed_kmh in TRAVEL_SPEEDS_KMH.items():
            seconds = (distance_km / speed_kmh) * 3600.0
            row[mode_name] = format_duration(seconds)

        summary_rows.append(row)

    return route_results, pd.DataFrame(summary_rows)



def calculate_routes_from_origin_to_destinations(
    origin_id,
    destination_ids,
):
    """Calculate one independent shortest path from one origin to each destination."""

    origin_id = int(origin_id)
    destination_ids = list(
        dict.fromkeys(int(fid) for fid in destination_ids)
    )

    if not destination_ids:
        raise ValueError("Select at least one destination building.")

    valid_ids = set(buildings_m["FID"].astype(int).tolist())
    requested_ids = [origin_id] + destination_ids
    missing_ids = [fid for fid in requested_ids if fid not in valid_ids]

    if missing_ids:
        raise ValueError(f"Building IDs not found: {missing_ids}")

    destination_ids = [
        fid for fid in destination_ids
        if fid != origin_id
    ]

    if not destination_ids:
        raise ValueError(
            "Destinations must be different from the origin."
        )

    route_results = []
    summary_rows = []

    for route_number, destination_id in enumerate(
        destination_ids,
        start=1,
    ):
        route, distance_m, origin_snap, destination_snap = (
            shortest_route_between_buildings(
                origin_id,
                destination_id,
            )
        )

        colour = MULTI_ROUTE_COLOURS[
            (route_number - 1) % len(MULTI_ROUTE_COLOURS)
        ]

        route = route.copy()
        route["origin_id"] = origin_id
        route["destination_id"] = destination_id
        route["route_label"] = f"{origin_id} → {destination_id}"
        route["route_colour"] = colour
        route["distance_m"] = float(distance_m)

        route_results.append(
            {
                "origin_id": origin_id,
                "destination_id": destination_id,
                "route": route,
                "distance_m": float(distance_m),
                "colour": colour,
            }
        )

        row = build_standard_route_row(
            origin_id=origin_id,
            destination_id=destination_id,
            distance_m=distance_m,
        )
        summary_rows.append(row)

    return route_results, pd.DataFrame(
        summary_rows,
        columns=STANDARD_CSV_COLUMNS,
    )




@st.cache_data(show_spinner=False)
def map_buildings_to_network_nodes(_buildings_projected):
    """
    Snap every building representative point to the nearest node in
    the main road network. The underscore prevents Streamlit from
    hashing the GeoDataFrame.
    """
    records = []

    for _, row in _buildings_projected.iterrows():
        building_id = int(row["FID"])
        point = row.geometry.representative_point()
        node, snap_distance = nearest_main_node(point)

        records.append(
            {
                "FID": building_id,
                "network_node": node,
                "snap_distance_m": float(snap_distance),
            }
        )

    return pd.DataFrame(records)


BUILDING_NODE_LOOKUP = map_buildings_to_network_nodes(buildings_m)


def calculate_service_areas(
    origin_id,
    minutes,
    travel_modes,
):
    """
    Calculate network-based service areas from one building.

    Each selected travel mode uses the same road graph but a different
    distance cutoff derived from the assumed average speed.
    """
    origin_id = int(origin_id)
    minutes = float(minutes)
    travel_modes = list(dict.fromkeys(travel_modes))

    if minutes <= 0:
        raise ValueError("Travel time must be greater than zero minutes.")

    if not travel_modes:
        raise ValueError("Select at least one travel mode.")

    invalid_modes = [
        mode for mode in travel_modes
        if mode not in TRAVEL_SPEEDS_KMH
    ]
    if invalid_modes:
        raise ValueError(f"Unknown travel modes: {invalid_modes}")

    origin_row = buildings_m[
        buildings_m["FID"] == origin_id
    ]
    if origin_row.empty:
        raise ValueError(f"Building {origin_id} was not found.")

    origin_point = origin_row.geometry.representative_point().iloc[0]
    origin_node, origin_snap_distance = nearest_main_node(origin_point)

    service_results = []
    summary_rows = []

    building_nodes = BUILDING_NODE_LOOKUP.set_index("FID")

    for mode_name in travel_modes:
        speed_kmh = float(TRAVEL_SPEEDS_KMH[mode_name])
        maximum_distance_m = speed_kmh * 1000.0 * (minutes / 60.0)

        node_distances = nx.single_source_dijkstra_path_length(
            G_MAIN,
            source=origin_node,
            cutoff=maximum_distance_m,
            weight="weight",
        )

        reachable_nodes = set(node_distances.keys())

        road_segments = []
        for node_a, node_b, edge_data in G_MAIN.edges(data=True):
            if node_a in reachable_nodes and node_b in reachable_nodes:
                edge_geometry = edge_data.get("geometry")
                if edge_geometry is not None and not edge_geometry.is_empty:
                    road_segments.append(edge_geometry)

        reachable_network = gpd.GeoDataFrame(
            geometry=road_segments,
            crs=PROJECTED_CRS,
        )

        reachable_building_ids = []
        for building_id, building_data in building_nodes.iterrows():
            network_node = building_data["network_node"]

            if network_node not in node_distances:
                continue

            network_distance_m = float(node_distances[network_node])

            # Include the origin with zero network distance.
            if int(building_id) == origin_id:
                network_distance_m = 0.0

            estimated_seconds = (
                network_distance_m / 1000.0 / speed_kmh
            ) * 3600.0

            reachable_building_ids.append(int(building_id))

            summary_rows.append(
                {
                    "Origin building": origin_id,
                    "Origin name": get_building_name(origin_id),
                    "Reachable building": int(building_id),
                    "Reachable name": get_building_name(building_id),
                    "Travel mode": mode_name,
                    "Time limit (min)": round(minutes, 1),
                    "Maximum network distance (m)": round(
                        maximum_distance_m,
                        1,
                    ),
                    "Network distance (m)": round(
                        network_distance_m,
                        1,
                    ),
                    "Estimated travel time": format_duration(
                        estimated_seconds
                    ),
                }
            )

        reachable_buildings = buildings_m[
            buildings_m["FID"].isin(reachable_building_ids)
        ].copy()

        service_results.append(
            {
                "origin_id": origin_id,
                "minutes": minutes,
                "mode": mode_name,
                "speed_kmh": speed_kmh,
                "maximum_distance_m": maximum_distance_m,
                "colour": SERVICE_AREA_COLOURS[mode_name],
                "reachable_nodes": reachable_nodes,
                "reachable_network": reachable_network,
                "reachable_buildings": reachable_buildings,
                "reachable_count": len(reachable_building_ids),
                "origin_snap_distance_m": origin_snap_distance,
            }
        )

    summary_table = pd.DataFrame(summary_rows)

    if not summary_table.empty:
        summary_table = summary_table.sort_values(
            ["Travel mode", "Network distance (m)", "Reachable building"]
        ).reset_index(drop=True)

    return service_results, summary_table


def service_area_table_to_csv(service_area_table):
    return service_area_table.to_csv(
        index=False
    ).encode("utf-8-sig")


def parse_service_area_command(question):
    """
    Recognise examples such as:
    'Show buildings reachable within 5 minutes walking from Building 10.'
    'Compare 5-minute walking, e-bike and driving service areas from Building 10.'
    """
    text = question.lower().strip()

    service_phrases = [
        "service area",
        "isochrone",
        "reachable within",
        "can reach within",
        "within",
    ]

    if not any(phrase in text for phrase in service_phrases):
        return None

    minute_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:minute|minutes|min)\b",
        text,
    )
    building_match = re.search(
        r"(?:from|around|starting at)\s+building\s*(\d+)",
        text,
    )

    if building_match is None:
        building_matches = re.findall(r"building\s*(\d+)", text)
        building_id = int(building_matches[-1]) if building_matches else None
    else:
        building_id = int(building_match.group(1))

    if minute_match is None or building_id is None:
        return None

    minutes = float(minute_match.group(1))
    modes = []

    if any(word in text for word in ["walk", "walking", "pedestrian"]):
        modes.append("Walking")
    if any(word in text for word in ["e-bike", "ebike", "e bike", "bicycle", "bike"]):
        modes.append("E-bike")
    if any(word in text for word in ["motorcycle", "motorbike"]):
        modes.append("Motorcycle")
    if any(word in text for word in ["drive", "driving", "car"]):
        modes.append("Car driving")

    if not modes:
        modes = ["Walking"]

    return {
        "action": "service_area",
        "origin_id": building_id,
        "minutes": minutes,
        "travel_modes": modes,
        "reply": (
            f"Calculating {minutes:g}-minute service areas from "
            f"Building {building_id} for {', '.join(modes)}."
        ),
    }


def parse_one_origin_multi_destination_command(question):
    """
    Recognise explicit one-origin to multiple-destination requests.

    Example:
    'Show separate shortest paths from Building 10 to Buildings 20, 30 and 40.'
    """
    text = question.lower().strip()
    numbers = [int(value) for value in re.findall(r"\d+", text)]

    explicit_pattern = re.search(
        r"from\s+building\s*(\d+).*?to\s+buildings\s+(.+)",
        text,
    )

    explicit_words = [
        "one origin",
        "multiple destinations",
        "separate destinations",
        "each destination",
    ]

    if len(numbers) >= 3 and (
        explicit_pattern is not None
        or any(phrase in text for phrase in explicit_words)
    ):
        return {
            "action": "one_origin_multi_destination",
            "origin_id": numbers[0],
            "destination_ids": numbers[1:],
            "reply": (
                f"Calculating separate shortest paths from Building "
                f"{numbers[0]} to Buildings {numbers[1:]}."
            ),
        }

    return None


def parse_multi_origin_command(question):
    """
    Recognise explicit multiple-origin to one-destination requests.

    Example:
    'Show separate shortest paths from Buildings 1, 2, 3 and 4
    to Building 444.'
    """
    text = question.lower().strip()
    numbers = [int(value) for value in re.findall(r"\d+", text)]

    explicit_pattern = re.search(
        r"from\s+buildings\s+(.+?)\s+to\s+building\s*(\d+)",
        text,
    )

    explicit_words = [
        "multiple origins",
        "each origin",
        "separate origins",
        "all origins",
    ]

    if len(numbers) >= 3 and (
        explicit_pattern is not None
        or any(phrase in text for phrase in explicit_words)
    ):
        return {
            "action": "multi_origin_route",
            "origin_ids": numbers[:-1],
            "destination_id": numbers[-1],
            "reply": (
                f"Calculating separate shortest paths from Buildings "
                f"{numbers[:-1]} to Building {numbers[-1]}."
            ),
        }

    return None


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


def format_duration(total_seconds):
    total_seconds = max(0, int(round(total_seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours} hr {minutes} min {seconds} sec"
    if minutes > 0:
        return f"{minutes} min {seconds} sec"
    return f"{seconds} sec"



STANDARD_CSV_COLUMNS = [
    "From building",
    "From name",
    "To building",
    "To name",
    "Distance (m)",
    "Walking",
    "E-bike",
    "Motorcycle",
    "Car driving",
]


def build_standard_route_row(
    origin_id,
    destination_id,
    distance_m,
):
    """Create one standard route-result row used by every route mode."""

    distance_m = float(distance_m)
    distance_km = distance_m / 1000.0

    row = {
        "From building": int(origin_id),
        "From name": get_building_name(origin_id),
        "To building": int(destination_id),
        "To name": get_building_name(destination_id),
        "Distance (m)": round(distance_m, 1),
    }

    for mode_name, speed_kmh in TRAVEL_SPEEDS_KMH.items():
        travel_seconds = (
            distance_km / speed_kmh
        ) * 3600.0
        row[mode_name] = format_duration(
            travel_seconds
        )

    return row


def standardise_route_table(route_legs):
    """
    Convert single, point-to-point and ordered multi-stop route legs
    into the same CSV structure used by independent routes.
    """

    rows = []

    for leg in route_legs or []:
        rows.append(
            build_standard_route_row(
                origin_id=leg["From building"],
                destination_id=leg["To building"],
                distance_m=leg["Distance (m)"],
            )
        )

    return pd.DataFrame(
        rows,
        columns=STANDARD_CSV_COLUMNS,
    )


def route_table_to_csv(route_table):
    """Return UTF-8 CSV bytes that open cleanly in Excel."""

    return route_table.to_csv(
        index=False,
    ).encode("utf-8-sig")


def add_direction_arrows(map_object, route_projected):
    """Add robust directional arrow markers without custom Leaflet plugins."""
    if route_projected is None or route_projected.empty:
        return

    arrow_group = folium.FeatureGroup(
        name="Route Direction Arrows",
        show=True,
    )

    # Add one arrow approximately every 70 metres of route.
    spacing_m = 70.0
    distance_since_arrow = 0.0

    for geometry in route_projected.geometry:
        if geometry is None or geometry.is_empty:
            continue

        coords = list(geometry.coords)
        if len(coords) < 2:
            continue

        segment_length = float(geometry.length)
        distance_since_arrow += segment_length

        if distance_since_arrow < spacing_m:
            continue

        distance_since_arrow = 0.0

        start_x, start_y = coords[0]
        end_x, end_y = coords[-1]
        angle = math.degrees(
            math.atan2(end_y - start_y, end_x - start_x)
        )

        midpoint_projected = geometry.interpolate(0.5, normalized=True)
        midpoint_wgs = (
            gpd.GeoSeries([midpoint_projected], crs=PROJECTED_CRS)
            .to_crs(WEB_CRS)
            .iloc[0]
        )

        folium.Marker(
            location=[midpoint_wgs.y, midpoint_wgs.x],
            tooltip="Route direction",
            icon=folium.DivIcon(
                icon_size=(30, 30),
                icon_anchor=(15, 15),
                html=f"""
                <div style="
                    width:30px;
                    height:30px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    transform:rotate({-angle}deg);
                    color:#FFFFFF;
                    font-size:24px;
                    font-weight:900;
                    text-shadow:
                        -2px -2px 0 {ROUTE_COLOUR},
                         2px -2px 0 {ROUTE_COLOUR},
                        -2px  2px 0 {ROUTE_COLOUR},
                         2px  2px 0 {ROUTE_COLOUR};
                "
                >➤</div>
                """,
            ),
        ).add_to(arrow_group)

    arrow_group.add_to(map_object)


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
    multi_route_results=None,
    multi_route_destination=None,
    service_area_results=None,
    service_area_origin=None,
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

    # Network service area / isochrone results
    if service_area_results:
        all_service_bounds = []

        for result in service_area_results:
            mode_name = result["mode"]
            colour = result["colour"]
            minutes = float(result["minutes"])
            maximum_distance_m = float(result["maximum_distance_m"])
            reachable_network = result["reachable_network"]
            reachable_buildings = result["reachable_buildings"]

            mode_group = folium.FeatureGroup(
                name=(
                    f"{mode_name}: {minutes:g}-minute service area"
                ),
                show=True,
            )

            if (
                reachable_network is not None
                and not reachable_network.empty
            ):
                network_wgs = reachable_network.to_crs(WEB_CRS)
                all_service_bounds.append(network_wgs.total_bounds)

                folium.GeoJson(
                    network_wgs,
                    style_function=(
                        lambda feature, c=colour: {
                            "color": c,
                            "weight": 6,
                            "opacity": 0.72,
                        }
                    ),
                    tooltip=folium.Tooltip(
                        f"{mode_name} reachable road network<br>"
                        f"Time limit: {minutes:g} minutes<br>"
                        f"Distance cutoff: {maximum_distance_m:.1f} m",
                        sticky=True,
                    ),
                ).add_to(mode_group)

            if (
                reachable_buildings is not None
                and not reachable_buildings.empty
            ):
                reachable_buildings_wgs = (
                    reachable_buildings.to_crs(WEB_CRS)
                )

                folium.GeoJson(
                    reachable_buildings_wgs,
                    style_function=(
                        lambda feature, c=colour: {
                            "color": c,
                            "weight": 2.5,
                            "fillColor": c,
                            "fillOpacity": 0.45,
                        }
                    ),
                    highlight_function=(
                        lambda feature, c=colour: {
                            "color": c,
                            "weight": 4,
                            "fillOpacity": 0.68,
                        }
                    ),
                    tooltip=folium.GeoJsonTooltip(
                        fields=["FID", "NAME"],
                        aliases=[
                            "Reachable building:",
                            "Building name:",
                        ],
                        sticky=True,
                    ),
                ).add_to(mode_group)

            mode_group.add_to(campus_map)

        origin_id = int(service_area_origin)
        origin_row = buildings_wgs[
            buildings_wgs["FID"] == origin_id
        ]

        if not origin_row.empty:
            origin_point = (
                origin_row.geometry
                .representative_point()
                .iloc[0]
            )

            origin_label = (
                f"Service-area origin {origin_id}: "
                f"{get_building_name(origin_id)}"
            )

            folium.Marker(
                [origin_point.y, origin_point.x],
                tooltip=origin_label,
                popup=origin_label,
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        background:#111827;
                        color:#FFFFFF;
                        border-radius:50%;
                        width:36px;
                        height:36px;
                        text-align:center;
                        line-height:36px;
                        font-weight:bold;
                        border:3px solid white;
                        box-shadow:0 1px 7px rgba(0,0,0,.75);
                    ">
                        {origin_id}
                    </div>
                    """
                ),
            ).add_to(campus_map)

        if all_service_bounds:
            min_x = min(bounds[0] for bounds in all_service_bounds)
            min_y = min(bounds[1] for bounds in all_service_bounds)
            max_x = max(bounds[2] for bounds in all_service_bounds)
            max_y = max(bounds[3] for bounds in all_service_bounds)

            campus_map.fit_bounds(
                [[min_y, min_x], [max_y, max_x]],
                padding=(45, 45),
            )

    # Multiple independent routes: A → Z, B → Z, C → Z, etc.
    elif multi_route_results:
        all_route_bounds = []

        for result in multi_route_results:
            origin_id = int(result["origin_id"])
            destination_id = int(result["destination_id"])
            distance_m = float(result["distance_m"])
            colour = result["colour"]
            route_projected = result["route"]
            route_wgs = route_projected.to_crs(WEB_CRS)

            all_route_bounds.append(route_wgs.total_bounds)

            route_name = f"Route {origin_id} → {destination_id}"
            tooltip = (
                f"<b>{get_building_name(origin_id)} → "
                f"{get_building_name(destination_id)}</b><br>"
                f"Distance: {distance_m:.1f} m"
            )

            folium.GeoJson(
                route_wgs,
                name=route_name,
                style_function=(
                    lambda feature, c=colour: {
                        "color": c,
                        "weight": 7,
                        "opacity": 0.95,
                    }
                ),
                tooltip=folium.Tooltip(tooltip, sticky=True),
            ).add_to(campus_map)

            # Direction arrows in the route's own colour.
            original_route_colour = globals()["ROUTE_COLOUR"]
            globals()["ROUTE_COLOUR"] = colour
            add_direction_arrows(campus_map, route_projected)
            globals()["ROUTE_COLOUR"] = original_route_colour

            origin_row = buildings_wgs[buildings_wgs["FID"] == origin_id]
            if not origin_row.empty:
                origin_point = origin_row.geometry.representative_point().iloc[0]
                label = f"Origin {origin_id}: {get_building_name(origin_id)}"
                folium.Marker(
                    [origin_point.y, origin_point.x],
                    tooltip=label,
                    popup=label,
                    icon=folium.DivIcon(html=f"""
                        <div style="background:{colour};color:white;border-radius:50%;
                        width:30px;height:30px;text-align:center;line-height:30px;
                        font-weight:bold;border:2px solid white;box-shadow:0 1px 5px rgba(0,0,0,.6);">
                        {origin_id}</div>
                    """),
                ).add_to(campus_map)

        # Add every unique destination marker. This works for both:
        # multiple origins → one destination, and
        # one origin → multiple destinations.
        destination_ids = list(
            dict.fromkeys(
                int(result["destination_id"])
                for result in multi_route_results
            )
        )

        for destination_id in destination_ids:
            destination_row = buildings_wgs[
                buildings_wgs["FID"] == destination_id
            ]

            if destination_row.empty:
                continue

            destination_point = (
                destination_row.geometry
                .representative_point()
                .iloc[0]
            )

            destination_label = (
                f"Destination {destination_id}: "
                f"{get_building_name(destination_id)}"
            )

            folium.Marker(
                [destination_point.y, destination_point.x],
                tooltip=destination_label,
                popup=destination_label,
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        background:#111827;
                        color:#FDE047;
                        border-radius:8px;
                        min-width:38px;
                        height:32px;
                        padding:0 7px;
                        text-align:center;
                        line-height:32px;
                        font-weight:bold;
                        border:2px solid white;
                        box-shadow:0 1px 6px rgba(0,0,0,.7);
                    ">
                        D:{destination_id}
                    </div>
                    """
                ),
            ).add_to(campus_map)

        if all_route_bounds:
            min_x = min(bounds[0] for bounds in all_route_bounds)
            min_y = min(bounds[1] for bounds in all_route_bounds)
            max_x = max(bounds[2] for bounds in all_route_bounds)
            max_y = max(bounds[3] for bounds in all_route_bounds)
            campus_map.fit_bounds(
                [[min_y, min_x], [max_y, max_x]],
                padding=(45, 45),
            )

    # Single or ordered multi-stop route result
    elif (
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

        # Add arrows that follow the actual ordered route segments.
        add_direction_arrows(
            map_object=campus_map,
            route_projected=route_result,
        )

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
                    f"Start: "
                    f"{get_building_name(building_id)}"
                )

            elif stop_number == len(
                building_ids
            ):
                colour = DESTINATION_COLOUR
                label = (
                    f"Destination: "
                    f"{get_building_name(building_id)}"
                )

            else:
                colour = INTERMEDIATE_COLOUR
                label = (
                    f"Stop {stop_number}: "
                    f"{get_building_name(building_id)}"
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
                        color:#111111;
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
    "Ask the AI GIS Assistant to run routing and service-area analyses. "
    "Manual tools remain available below in collapsible panels."
)




# ============================================================
# 17. GEMINI AI GIS ASSISTANT
# ============================================================

st.subheader("🤖 AI GIS Assistant")

st.caption(
    "Enter a natural-language GIS request below. Gemini interprets the "
    "request, while the GIS engine performs routing or service-area analysis."
)

with st.form(
    key="gis_command_form",
    clear_on_submit=False,
):
    question = st.text_input(
        "Enter your GIS request",
        value=st.session_state.last_question,
        placeholder=(
            "Example: Find the shortest route from Building 10 "
            "to Building 20"
        ),
    )

    with st.expander(
        "📖 Example GIS commands",
        expanded=False,
    ):
        st.markdown(
            """
            **One point → one point**  
            `Find the shortest route from Building 10 to Building 20.`

            **Ordered multi-stop route**  
            `Start at Building 10, visit Buildings 20 and 35, then go to Building 50.`

            **Multiple Origins → One Destination**  
            `Show separate shortest paths from Buildings 1, 2, 3 and 4 to Building 444.`

            **One Origin → Multiple Destinations**  
            `Show separate shortest paths from Building 10 to Buildings 20, 30 and 40.`

            **Service Area — one mode**  
            `Show every building reachable within 5 minutes walking from Building 10.`

            **Service Area — compare modes**  
            `Compare 5-minute walking, e-bike and car driving service areas from Building 10.`
            """
        )

    run_col, clear_col = st.columns([3, 1])

    with run_col:
        submitted = st.form_submit_button(
            "▶ Run AI GIS Analysis",
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        clear_pressed = st.form_submit_button(
            "🗑️ Clear",
            use_container_width=True,
        )


if clear_pressed:
    st.session_state.route_result = None
    st.session_state.route_legs = None
    st.session_state.total_distance = None
    st.session_state.selected_building_ids = None

    st.session_state.multi_route_results = None
    st.session_state.multi_route_table = None
    st.session_state.multi_route_destination = None
    st.session_state.independent_route_mode = None

    st.session_state.service_area_results = None
    st.session_state.service_area_table = None
    st.session_state.service_area_origin = None
    st.session_state.service_area_minutes = None

    st.session_state.last_question = ""
    st.session_state.last_ai_reply = ""
    st.session_state.last_interpreter = ""

    st.rerun()


if submitted:
    st.session_state.last_question = question

    if not question.strip():
        st.warning("Please enter a GIS request.")

    else:
        parsed = None
        interpreter_used = ""

        # --------------------------------------------------------
        # AI-FIRST INTERPRETATION
        # Gemini always receives the request first. The built-in
        # parsers are used only when Gemini is unavailable or fails.
        # --------------------------------------------------------
        try:
            gemini_key = str(
                st.secrets.get("GEMINI_API_KEY", "")
            ).strip()

            if not gemini_key:
                raise ValueError(
                    "GEMINI_API_KEY is missing from Streamlit Secrets."
                )

            with st.spinner("Gemini is interpreting your GIS request..."):
                parsed = interpret_gis_command(
                    question=question,
                    api_key=gemini_key,
                )

            interpreter_used = "Gemini"

            # Compatibility safeguard: an older ai_engine.py may return
            # action=unknown with a refusal for service-area requests.
            # In that case, recover with the built-in parser instead of
            # showing a false unsupported-analysis message.
            if not isinstance(parsed, dict):
                parsed = {"action": "unknown", "reply": ""}

            if parsed.get("action") == "unknown":
                fallback_parsed = parse_service_area_command(question)

                if fallback_parsed is None:
                    fallback_parsed = (
                        parse_one_origin_multi_destination_command(question)
                    )

                if fallback_parsed is None:
                    fallback_parsed = parse_multi_origin_command(question)

                if fallback_parsed is None:
                    fallback_parsed = parse_command(question)

                if fallback_parsed.get("action") != "unknown":
                    parsed = fallback_parsed
                    interpreter_used = (
                        "Built-in fallback after Gemini returned unknown"
                    )

        except Exception as gemini_error:
            # ----------------------------------------------------
            # FALLBACK PARSERS
            # These keep the dashboard usable if Gemini is offline,
            # quota-limited, misconfigured or temporarily unavailable.
            # ----------------------------------------------------
            parsed = parse_service_area_command(question)

            if parsed is not None:
                interpreter_used = "Built-in service-area fallback"
            else:
                parsed = parse_one_origin_multi_destination_command(
                    question
                )

                if parsed is not None:
                    interpreter_used = (
                        "Built-in one-origin/multiple-destination fallback"
                    )
                else:
                    parsed = parse_multi_origin_command(question)

                    if parsed is not None:
                        interpreter_used = (
                            "Built-in multiple-origin fallback"
                        )
                    else:
                        parsed = parse_command(question)
                        interpreter_used = "Rule-based route fallback"

            st.warning(
                "Gemini could not interpret the request, so the dashboard "
                "used its built-in fallback parser. "
                f"Details: {gemini_error}"
            )

        if parsed is None:
            parsed = {
                "action": "unknown",
                "reply": "The GIS request could not be interpreted.",
            }

        st.session_state.last_interpreter = interpreter_used
        st.session_state.last_ai_reply = str(
            parsed.get("reply", "")
        ).strip()

        if parsed.get("action") == "service_area":
            try:
                origin_id = int(parsed.get("origin_id"))
                minutes = float(parsed.get("minutes"))
                travel_modes = list(
                    parsed.get("travel_modes", ["Walking"])
                )

                with st.spinner(
                    "Calculating GIS network service areas..."
                ):
                    service_results, service_table = (
                        calculate_service_areas(
                            origin_id=origin_id,
                            minutes=minutes,
                            travel_modes=travel_modes,
                        )
                    )

                st.session_state.service_area_results = service_results
                st.session_state.service_area_table = service_table
                st.session_state.service_area_origin = origin_id
                st.session_state.service_area_minutes = minutes

                st.session_state.route_result = None
                st.session_state.route_legs = None
                st.session_state.total_distance = None
                st.session_state.selected_building_ids = None
                st.session_state.multi_route_results = None
                st.session_state.multi_route_table = None
                st.session_state.multi_route_destination = None
                st.session_state.independent_route_mode = None

                if st.session_state.last_ai_reply:
                    st.info(st.session_state.last_ai_reply)

                st.success(
                    f"Calculated {len(service_results)} network "
                    f"service area(s) from Building {origin_id}."
                )

            except Exception as error:
                st.error(
                    f"Unable to calculate service area: {error}"
                )

        elif parsed.get("action") == "one_origin_multi_destination":
            try:
                origin_id = int(
                    parsed.get("origin_id")
                )

                destination_ids = [
                    int(fid)
                    for fid in parsed.get(
                        "destination_ids",
                        [],
                    )
                ]

                with st.spinner(
                    "Calculating separate GIS shortest paths..."
                ):
                    multi_results, multi_table = (
                        calculate_routes_from_origin_to_destinations(
                            origin_id=origin_id,
                            destination_ids=destination_ids,
                        )
                    )

                st.session_state.multi_route_results = multi_results
                st.session_state.multi_route_table = multi_table
                st.session_state.multi_route_destination = None
                st.session_state.independent_route_mode = (
                    "one_origin_multiple_destinations"
                )

                st.session_state.route_result = None
                st.session_state.route_legs = None
                st.session_state.total_distance = None
                st.session_state.selected_building_ids = None
                st.session_state.service_area_results = None
                st.session_state.service_area_table = None
                st.session_state.service_area_origin = None
                st.session_state.service_area_minutes = None

                if st.session_state.last_ai_reply:
                    st.info(
                        st.session_state.last_ai_reply
                    )

                st.success(
                    f"Calculated {len(multi_results)} independent routes "
                    f"from Building {origin_id}."
                )

            except Exception as error:
                st.error(
                    f"Unable to calculate routes: {error}"
                )

        elif parsed.get("action") == "multi_origin_route":
            try:
                origin_ids = [int(fid) for fid in parsed.get("origin_ids", [])]
                destination_id = int(parsed.get("destination_id"))

                with st.spinner("Calculating separate GIS shortest paths..."):
                    multi_results, multi_table = (
                        calculate_routes_from_origins_to_destination(
                            origin_ids=origin_ids,
                            destination_id=destination_id,
                        )
                    )

                st.session_state.multi_route_results = multi_results
                st.session_state.multi_route_table = multi_table
                st.session_state.multi_route_destination = destination_id
                st.session_state.independent_route_mode = (
                    "multiple_origins_one_destination"
                )
                st.session_state.route_result = None
                st.session_state.route_legs = None
                st.session_state.total_distance = None
                st.session_state.selected_building_ids = None
                st.session_state.service_area_results = None
                st.session_state.service_area_table = None
                st.session_state.service_area_origin = None
                st.session_state.service_area_minutes = None

                if st.session_state.last_ai_reply:
                    st.info(st.session_state.last_ai_reply)

                st.success(
                    f"Calculated {len(multi_results)} independent routes "
                    f"to Building {destination_id}."
                )

            except Exception as error:
                st.error(f"Unable to calculate routes: {error}")

        elif parsed.get("action") == "route":
            building_ids = [
                int(building_id)
                for building_id in parsed.get(
                    "building_ids",
                    [],
                )
            ]

            if len(building_ids) < 2:
                st.warning(
                    "Please provide at least two building FIDs."
                )

            else:
                try:
                    with st.spinner(
                        "Running GIS shortest-path analysis..."
                    ):
                        (
                            route_result,
                            route_legs,
                            total_distance,
                        ) = calculate_multi_stop_route(
                            building_ids
                        )

                    st.session_state.route_result = route_result
                    st.session_state.route_legs = route_legs
                    st.session_state.total_distance = total_distance
                    st.session_state.selected_building_ids = (
                        building_ids
                    )
                    st.session_state.multi_route_results = None
                    st.session_state.multi_route_table = None
                    st.session_state.multi_route_destination = None
                    st.session_state.independent_route_mode = None
                    st.session_state.service_area_results = None
                    st.session_state.service_area_table = None
                    st.session_state.service_area_origin = None
                    st.session_state.service_area_minutes = None

                    if st.session_state.last_ai_reply:
                        st.info(
                            st.session_state.last_ai_reply
                        )

                    st.success(
                        "Route calculated successfully using "
                        f"{interpreter_used} for language interpretation "
                        "and the GIS engine for network analysis."
                    )

                except Exception as error:
                    st.error(
                        "Unable to calculate route: "
                        f"{error}"
                    )

        else:
            if st.session_state.last_ai_reply:
                st.info(
                    st.session_state.last_ai_reply
                )

            st.warning(
                "The request was not recognised as a supported GIS request. "
                "Try a shortest-path, multi-route or service-area example "
                "shown above."
            )


if st.session_state.last_interpreter:
    st.caption(
        f"Last language interpreter used: "
        f"{st.session_state.last_interpreter}"
    )



# ============================================================
# COLLAPSIBLE MANUAL GIS TOOLS
# ============================================================

st.markdown("### Manual GIS Tools")
st.caption(
    "Open only the tool you need. Click the arrow again to collapse it."
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
# COLLAPSIBLE: 🧭 Manual Building Route Planner
# ============================================================

with st.expander("🧭 Manual Building Route Planner", expanded=False):
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
            st.session_state.multi_route_results = None
            st.session_state.multi_route_table = None
            st.session_state.multi_route_destination = None
            st.session_state.independent_route_mode = None
            st.session_state.service_area_results = None
            st.session_state.service_area_table = None
            st.session_state.service_area_origin = None
            st.session_state.service_area_minutes = None
            st.success("Route calculated successfully.")

        except Exception as error:
            st.error(f"Unable to calculate route: {error}")

# ============================================================
# COLLAPSIBLE: 🔀 Multiple Origins → One Destination
# ============================================================

with st.expander("🔀 Multiple Origins → One Destination", expanded=False):
    st.caption(
        "Calculate a separate shortest path from every selected origin "
        "to the same destination. Example: 1 → 444, 2 → 444, "
        "3 → 444 and 4 → 444."
    )

    multi_col1, multi_col2 = st.columns([2, 1])

    with multi_col1:
        multi_origin_labels = st.multiselect(
            "Origin buildings",
            options=building_labels,
            key="multi_origin_buildings_select",
        )

    with multi_col2:
        multi_destination_label = st.selectbox(
            "Common destination",
            options=building_labels,
            key="multi_destination_building_select",
        )

    if st.button(
        "Calculate separate routes to destination",
        type="primary",
        key="calculate_multi_origin_routes_button",
    ):
        try:
            origin_ids = [building_label_to_fid[label] for label in multi_origin_labels]
            destination_id = building_label_to_fid[multi_destination_label]

            with st.spinner("Calculating separate shortest paths..."):
                multi_results, multi_table = (
                    calculate_routes_from_origins_to_destination(
                        origin_ids=origin_ids,
                        destination_id=destination_id,
                    )
                )

            st.session_state.multi_route_results = multi_results
            st.session_state.multi_route_table = multi_table
            st.session_state.multi_route_destination = destination_id
            st.session_state.independent_route_mode = "multiple_origins_one_destination"

            # Clear the previous single/ordered route so the modes do not overlap.
            st.session_state.route_result = None
            st.session_state.route_legs = None
            st.session_state.total_distance = None
            st.session_state.selected_building_ids = None
            st.session_state.service_area_results = None
            st.session_state.service_area_table = None
            st.session_state.service_area_origin = None
            st.session_state.service_area_minutes = None

            st.success(
                f"Calculated {len(multi_results)} separate shortest paths "
                f"to Building {destination_id}."
            )

        except Exception as error:
            st.error(f"Unable to calculate the separate routes: {error}")

# ============================================================
# COLLAPSIBLE: 🌐 One Origin → Multiple Destinations
# ============================================================

with st.expander("🌐 One Origin → Multiple Destinations", expanded=False):
    st.caption(
        "Calculate a separate shortest path from one origin to every "
        "selected destination. Example: 10 → 20, 10 → 30 and 10 → 40."
    )

    one_many_col1, one_many_col2 = st.columns([1, 2])

    with one_many_col1:
        one_origin_label = st.selectbox(
            "Common origin",
            options=building_labels,
            key="one_origin_building_select",
        )

    with one_many_col2:
        multiple_destination_labels = st.multiselect(
            "Destination buildings",
            options=building_labels,
            key="multiple_destination_buildings_select",
        )

    if st.button(
        "Calculate separate routes from origin",
        type="primary",
        key="calculate_one_origin_routes_button",
    ):
        try:
            origin_id = building_label_to_fid[
                one_origin_label
            ]

            destination_ids = [
                building_label_to_fid[label]
                for label in multiple_destination_labels
            ]

            with st.spinner(
                "Calculating separate shortest paths..."
            ):
                multi_results, multi_table = (
                    calculate_routes_from_origin_to_destinations(
                        origin_id=origin_id,
                        destination_ids=destination_ids,
                    )
                )

            st.session_state.multi_route_results = multi_results
            st.session_state.multi_route_table = multi_table
            st.session_state.multi_route_destination = None
            st.session_state.independent_route_mode = (
                "one_origin_multiple_destinations"
            )

            st.session_state.route_result = None
            st.session_state.route_legs = None
            st.session_state.total_distance = None
            st.session_state.selected_building_ids = None
            st.session_state.service_area_results = None
            st.session_state.service_area_table = None
            st.session_state.service_area_origin = None
            st.session_state.service_area_minutes = None

            st.success(
                f"Calculated {len(multi_results)} separate shortest paths "
                f"from Building {origin_id}."
            )

        except Exception as error:
            st.error(
                f"Unable to calculate the separate routes: {error}"
            )

# ============================================================
# COLLAPSIBLE: ⏱️ Service Area (Isochrone)
# ============================================================

with st.expander("⏱️ Service Area (Isochrone)", expanded=False):
    st.caption(
        "Select one starting building, a time limit and one or more travel "
        "modes. The GIS engine highlights the road network and buildings "
        "reachable within that time. Each travel mode uses its own assumed "
        "average speed."
    )

    service_col1, service_col2, service_col3 = st.columns([2, 1, 2])

    with service_col1:
        service_origin_label = st.selectbox(
            "Starting building",
            options=building_labels,
            key="service_area_origin_select",
        )

    with service_col2:
        service_minutes = st.number_input(
            "Travel-time limit (minutes)",
            min_value=1.0,
            max_value=120.0,
            value=5.0,
            step=1.0,
            key="service_area_minutes_input",
        )

    with service_col3:
        service_modes = st.multiselect(
            "Travel modes",
            options=list(TRAVEL_SPEEDS_KMH.keys()),
            default=["Walking"],
            key="service_area_modes_select",
        )

    st.caption(
        "Example comparison: select Walking, E-bike and Car driving with "
        "a 5-minute limit to compare how many buildings are reachable."
    )

    if st.button(
        "Calculate service area",
        type="primary",
        key="calculate_service_area_button",
    ):
        try:
            service_origin_id = building_label_to_fid[
                service_origin_label
            ]

            with st.spinner(
                "Calculating network service areas and reachable buildings..."
            ):
                service_results, service_table = calculate_service_areas(
                    origin_id=service_origin_id,
                    minutes=service_minutes,
                    travel_modes=service_modes,
                )

            st.session_state.service_area_results = service_results
            st.session_state.service_area_table = service_table
            st.session_state.service_area_origin = service_origin_id
            st.session_state.service_area_minutes = float(service_minutes)

            # Clear route outputs so different analyses do not overlap.
            st.session_state.route_result = None
            st.session_state.route_legs = None
            st.session_state.total_distance = None
            st.session_state.selected_building_ids = None
            st.session_state.multi_route_results = None
            st.session_state.multi_route_table = None
            st.session_state.multi_route_destination = None
            st.session_state.independent_route_mode = None

            st.success(
                f"Calculated {len(service_results)} service area(s) "
                f"from Building {service_origin_id}."
            )

        except Exception as error:
            st.error(f"Unable to calculate service area: {error}")


# ============================================================
# 16. DISPLAY MAP FIRST
# ============================================================

st.subheader("Route map")

campus_map = create_map(
    route_result=st.session_state.route_result,
    route_legs=st.session_state.route_legs,
    total_distance=st.session_state.total_distance,
    building_ids=(
        st.session_state
        .selected_building_ids
    ),
    multi_route_results=st.session_state.multi_route_results,
    multi_route_destination=st.session_state.multi_route_destination,
    service_area_results=st.session_state.service_area_results,
    service_area_origin=st.session_state.service_area_origin,
)

if st.session_state.service_area_results:
    map_ids = [
        "service",
        st.session_state.service_area_origin,
        st.session_state.service_area_minutes,
    ] + [
        result["mode"]
        for result in st.session_state.service_area_results
    ]
elif st.session_state.multi_route_results:
    map_ids = [
        result["origin_id"]
        for result in st.session_state.multi_route_results
    ] + [st.session_state.multi_route_destination]
else:
    map_ids = st.session_state.selected_building_ids or ["default"]

map_key = "main_route_map_" + "_".join(str(value) for value in map_ids)

st_folium(
    campus_map,
    width=None,
    height=720,
    returned_objects=[],
    use_container_width=True,
    key=map_key,
)


# ============================================================
# 19. SERVICE AREA RESULTS
# ============================================================

if (
    st.session_state.service_area_results is not None
    and st.session_state.service_area_table is not None
):
    st.subheader("Service area results")

    origin_id = int(st.session_state.service_area_origin)
    minutes = float(st.session_state.service_area_minutes)

    st.caption(
        f"Buildings reachable from Building {origin_id} within "
        f"{minutes:g} minutes using the selected travel modes. "
        "Distances are calculated along the road network."
    )

    metric_columns = st.columns(
        len(st.session_state.service_area_results)
    )

    for column, result in zip(
        metric_columns,
        st.session_state.service_area_results,
    ):
        with column:
            mode_name = result["mode"]
            st.metric(
                f"{SERVICE_AREA_ICONS[mode_name]} {mode_name}",
                f"{result['reachable_count']} buildings",
            )
            st.caption(
                f"Maximum network distance: "
                f"{result['maximum_distance_m']:.1f} m"
            )

    service_table = st.session_state.service_area_table.copy()

    selected_mode_filter = st.multiselect(
        "Filter result table by travel mode",
        options=[
            result["mode"]
            for result in st.session_state.service_area_results
        ],
        default=[
            result["mode"]
            for result in st.session_state.service_area_results
        ],
        key="service_area_result_mode_filter",
    )

    filtered_service_table = service_table[
        service_table["Travel mode"].isin(selected_mode_filter)
    ].copy()

    st.dataframe(
        filtered_service_table,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download service area results as CSV",
        data=service_area_table_to_csv(service_table),
        file_name=(
            f"service_area_building_{origin_id}_"
            f"{minutes:g}_minutes.csv"
        ),
        mime="text/csv",
        key="download_service_area_csv",
    )

    st.caption(
        "The current analysis uses assumed constant average speeds and "
        "the same road network for all modes. It does not yet enforce "
        "mode-specific restrictions such as pedestrian-only paths, "
        "vehicle access, stairs, traffic or junction delays."
    )


# ============================================================
# 20. MULTIPLE-ORIGIN ROUTE RESULTS
# ============================================================

if (
    st.session_state.multi_route_results is not None
    and st.session_state.multi_route_table is not None
):
    st.subheader("Separate route summary")

    if (
        st.session_state.independent_route_mode
        == "one_origin_multiple_destinations"
    ):
        st.caption(
            "Every row is an independent shortest path from the same "
            "origin to a different destination. The routes are not combined."
        )
    else:
        st.caption(
            "Every row is an independent shortest path from a different "
            "origin to the same destination. The routes are not combined "
            "and are not ranked."
        )

    standard_multi_table = (
        st.session_state.multi_route_table[
            STANDARD_CSV_COLUMNS
        ].copy()
    )

    st.dataframe(
        standard_multi_table,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download route results as CSV",
        data=route_table_to_csv(
            standard_multi_table
        ),
        file_name="route_results.csv",
        mime="text/csv",
        key="download_independent_routes_csv",
    )

    for result in st.session_state.multi_route_results:
        origin_id = result["origin_id"]
        destination_id = result["destination_id"]
        colour = result["colour"]
        distance_m = result["distance_m"]
        st.markdown(
            f"<span style='color:{colour};font-size:20px'>●</span> "
            f"**Building {origin_id} → Building {destination_id}:** "
            f"{distance_m:.1f} m",
            unsafe_allow_html=True,
        )


# ============================================================
# 19. ROUTE RESULTS AND TRAVEL TIMES
# ============================================================

if (
    st.session_state.route_result is not None
    and st.session_state.total_distance is not None
):
    total_distance = float(st.session_state.total_distance)
    distance_km = total_distance / 1000.0

    st.subheader("Route summary")

    result1, result2 = st.columns(2)

    with result1:
        st.metric(
            "Stops",
            len(st.session_state.selected_building_ids),
        )

    with result2:
        st.metric(
            "Total road distance",
            f"{total_distance:.1f} m",
        )

    time_columns = st.columns(4)
    icons = ["🚶", "🚲", "🏍️", "🚗"]

    for column, icon, (mode_name, speed_kmh) in zip(
        time_columns,
        icons,
        TRAVEL_SPEEDS_KMH.items(),
    ):
        travel_seconds = (distance_km / speed_kmh) * 3600.0

        with column:
            st.metric(
                f"{icon} {mode_name}",
                format_duration(travel_seconds),
            )
            st.caption(
                f"Assumed average speed: {speed_kmh:g} km/h"
            )

    st.caption(
        "Travel times are approximate and use the same road-network "
        "distance. They do not account for traffic, parking, road "
        "restrictions, junction delays or vehicle access."
    )

    route_table = pd.DataFrame(
        st.session_state.route_legs
    )

    st.dataframe(
        route_table,
        use_container_width=True,
        hide_index=True,
    )
