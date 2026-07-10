from pathlib import Path

import folium
import geopandas as gpd
import streamlit as st
from folium.plugins import Fullscreen
from streamlit_folium import st_folium


# ============================================================
# 1. STREAMLIT PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI GIS Dashboard",
    page_icon="🗺️",
    layout="wide"
)


# ============================================================
# 2. FILE AND WEB-LAYER SETTINGS
# ============================================================

APP_FOLDER = Path(__file__).parent

BUILDING_FILE = APP_FOLDER / "Building_FeaturesToJSON.geojson"
ROAD_FILE = APP_FOLDER / "ROAD_FeaturesToJSON.geojson"

ORTHO_TILE_URL = (
    "https://tiles.arcgis.com/tiles/2ZRAaoTSJbQ20ceg/"
    "arcgis/rest/services/UiTM_Shah_Alam_Orthomosaic/"
    "MapServer/tile/{z}/{y}/{x}"
)


# ============================================================
# 3. COLOUR SETTINGS
# ============================================================

BUILDING_COLOUR = "#F39C12"   # Orange
ROAD_COLOUR = "#8B4513"       # Chocolate brown


# ============================================================
# 4. LOAD GIS DATA
# ============================================================

@st.cache_data
def load_gis_data():
    if not BUILDING_FILE.exists():
        raise FileNotFoundError(
            f"Building file was not found: {BUILDING_FILE.name}"
        )

    if not ROAD_FILE.exists():
        raise FileNotFoundError(
            f"Road file was not found: {ROAD_FILE.name}"
        )

    buildings = gpd.read_file(BUILDING_FILE)
    roads = gpd.read_file(ROAD_FILE)

    if buildings.crs is None:
        raise ValueError("The building layer has no coordinate system.")

    if roads.crs is None:
        raise ValueError("The road layer has no coordinate system.")

    if "FID" not in buildings.columns:
        raise ValueError("The building layer does not contain an FID field.")

    buildings = buildings.to_crs(epsg=4326)
    roads = roads.to_crs(epsg=4326)

    return buildings, roads


try:
    buildings_wgs, roads_wgs = load_gis_data()

except Exception as error:
    st.error(f"Unable to load GIS data: {error}")
    st.stop()


# ============================================================
# 5. CALCULATE MAP CENTRE
# ============================================================

campus_geometry = buildings_wgs.geometry.union_all()
campus_centre = campus_geometry.centroid

centre_latitude = campus_centre.y
centre_longitude = campus_centre.x


# ============================================================
# 6. CREATE MAP
# ============================================================

def create_campus_map():
    campus_map = folium.Map(
        location=[centre_latitude, centre_longitude],
        zoom_start=17,
        tiles=None,
        max_zoom=23,
        control_scale=True
    )

    # OpenStreetMap can be switched on or off
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=True,
        control=True,
        show=False
    ).add_to(campus_map)

    # UiTM orthomosaic can be switched on or off
    folium.TileLayer(
        tiles=ORTHO_TILE_URL,
        name="UiTM Shah Alam Orthomosaic",
        attr="UiTM Shah Alam Orthomosaic",
        overlay=True,
        control=True,
        show=True,
        max_zoom=23,
        max_native_zoom=20
    ).add_to(campus_map)

    # Road network
    folium.GeoJson(
        roads_wgs,
        name="Road Network",
        style_function=lambda feature: {
            "color": ROAD_COLOUR,
            "weight": 3,
            "opacity": 0.8
        }
    ).add_to(campus_map)

    # Building footprints
    folium.GeoJson(
        buildings_wgs,
        name="Building Footprints",
        style_function=lambda feature: {
            "color": BUILDING_COLOUR,
            "weight": 1,
            "fillColor": BUILDING_COLOUR,
            "fillOpacity": 0.25
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["FID"],
            aliases=["Building ID:"],
            sticky=True
        )
    ).add_to(campus_map)

    # Full-screen button
    Fullscreen(
        position="topright",
        title="Full Screen",
        title_cancel="Exit Full Screen",
        force_separate_button=True
    ).add_to(campus_map)

    # Layer control on the left
    folium.LayerControl(
        collapsed=False,
        position="topleft"
    ).add_to(campus_map)

    return campus_map


# ============================================================
# 7. STREAMLIT INTERFACE
# ============================================================

st.title("🗺️ AI GIS Dashboard")

st.caption(
    "Interactive UiTM Shah Alam campus map using UAV orthomosaic, "
    "building footprints and road network."
)

column1, column2, column3 = st.columns(3)

with column1:
    st.metric("Buildings", f"{len(buildings_wgs):,}")

with column2:
    st.metric("Road features", f"{len(roads_wgs):,}")

with column3:
    st.metric("Current capability", "Map viewer")


st.info(
    "Use the layer control on the upper-left of the map to switch "
    "the orthomosaic, OpenStreetMap, roads and buildings on or off."
)

campus_map = create_campus_map()

st_folium(
    campus_map,
    width=None,
    height=720,
    returned_objects=[]
)
