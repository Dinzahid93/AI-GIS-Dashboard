from __future__ import annotations

from typing import Any, Optional

import folium
import streamlit as st
from folium.plugins import Fullscreen, SideBySideLayers
from streamlit_folium import st_folium


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


def show_terrain_analysis(
    orthophoto_tile_url: Optional[str] = None,
    buildings_geojson: Optional[dict[str, Any]] = None,
    roads_geojson: Optional[dict[str, Any]] = None,
) -> None:

    st.subheader("Terrain Analysis")

    terrain_mode = st.selectbox(
        "Terrain layer",
        [
            "DTM",
            "DSM",
            "DTM vs DSM Swipe",
        ],
    )

    terrain_opacity = st.slider(
        "Terrain opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.8,
        step=0.1,
    )

    show_orthophoto = st.toggle(
        "Show orthophoto",
        value=True,
    )

    terrain_map = folium.Map(
        location=[3.0697, 101.5033],
        zoom_start=16,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
        show=not show_orthophoto,
    ).add_to(terrain_map)

    if orthophoto_tile_url:
        folium.TileLayer(
            tiles=orthophoto_tile_url,
            name="Orthophoto",
            attr="UiTM Shah Alam Orthophoto",
            overlay=False,
            control=True,
            show=show_orthophoto,
        ).add_to(terrain_map)

    if terrain_mode == "DTM":
        folium.TileLayer(
            tiles=DTM_TILE_URL,
            name="DTM",
            attr="UiTM Shah Alam DTM",
            overlay=True,
            control=True,
            show=True,
            opacity=terrain_opacity,
        ).add_to(terrain_map)

    elif terrain_mode == "DSM":
        folium.TileLayer(
            tiles=DSM_TILE_URL,
            name="DSM",
            attr="UiTM Shah Alam DSM",
            overlay=True,
            control=True,
            show=True,
            opacity=terrain_opacity,
        ).add_to(terrain_map)

    elif terrain_mode == "DTM vs DSM Swipe":
        dtm_layer = folium.TileLayer(
            tiles=DTM_TILE_URL,
            name="DTM",
            attr="UiTM Shah Alam DTM",
            overlay=True,
            opacity=terrain_opacity,
        )

        dsm_layer = folium.TileLayer(
            tiles=DSM_TILE_URL,
            name="DSM",
            attr="UiTM Shah Alam DSM",
            overlay=True,
            opacity=terrain_opacity,
        )

        dtm_layer.add_to(terrain_map)
        dsm_layer.add_to(terrain_map)

        SideBySideLayers(
            layer_left=dtm_layer,
            layer_right=dsm_layer,
        ).add_to(terrain_map)

    Fullscreen(
        position="topleft",
    ).add_to(terrain_map)

    folium.LayerControl(
        collapsed=False,
    ).add_to(terrain_map)

    st_folium(
        terrain_map,
        width=None,
        height=650,
        key=f"terrain_map_{terrain_mode}_{terrain_opacity}",
    )
