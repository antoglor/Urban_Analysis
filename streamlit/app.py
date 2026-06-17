# app.py
import streamlit as st
from streamlit_folium import st_folium
import ee
import folium
from google.oauth2 import service_account
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# EARTH ENGINE AUTH
# -----------------------------
sa = st.secrets["google_service_account"]

credentials = service_account.Credentials.from_service_account_info(
    dict(sa),
    scopes=["https://www.googleapis.com/auth/earthengine"]
)

try:
    ee.Initialize(credentials)
except Exception:
    ee.Authenticate()
    ee.Initialize()

# -----------------------------
# EE -> FOLIUM HELPER
# -----------------------------
def add_ee_layer(self, ee_image, vis_params, name):
    map_id = ee.Image(ee_image).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
    ).add_to(self)

folium.Map.add_ee_layer = add_ee_layer

# -----------------------------
# PAGE
# -----------------------------
st.set_page_config(layout="wide")
st.title("Urban Analysis Tool")
st.write("Select a point and run analysis.")

if "clicked_lat" not in st.session_state:
    st.session_state.clicked_lat = None

if "clicked_lon" not in st.session_state:
    st.session_state.clicked_lon = None

# -----------------------------
# SELECTOR MAP
# -----------------------------
selector_map = folium.Map(
    location=[28.6, 77.2],
    zoom_start=5,
    tiles="CartoDB positron"
)

if st.session_state.clicked_lat:
    folium.Circle(
        location=[
            st.session_state.clicked_lat,
            st.session_state.clicked_lon
        ],
        radius=500,
        color="blue",
        fill=True
    ).add_to(selector_map)

map_data = st_folium(selector_map, height=350, width=800)

if map_data and map_data.get("last_clicked"):
    st.session_state.clicked_lat = map_data["last_clicked"]["lat"]
    st.session_state.clicked_lon = map_data["last_clicked"]["lng"]

col1, col2 = st.columns(2)

with col1:
    st.write("Latitude:", st.session_state.clicked_lat)

with col2:
    st.write("Longitude:", st.session_state.clicked_lon)

# -----------------------------
# ANALYSIS
# -----------------------------
if st.button("Run Analysis"):

    if st.session_state.clicked_lat is None:
        st.error("Select a point first.")
        st.stop()

    lat = st.session_state.clicked_lat
    lon = st.session_state.clicked_lon

    with st.spinner("Running Earth Engine analysis..."):
        st.session_state["results"] = run_analysis(lat, lon)

        roi = ee.Geometry.Point([lon, lat]).buffer(1000)

        buildings = (
            ee.FeatureCollection(
                "projects/sat-io/open-datasets/MSBuildings/India"
            )
            .filterBounds(roi)
        )

        buildings = buildings.filter(
            ee.Filter.gte("area", 16)
        )

        open_buildings = (
            ee.ImageCollection(
                "GOOGLE/Research/open-buildings-temporal/v1"
            )
            .filterBounds(roi)
            .select("building_height")
            .median()
            .clip(roi)
        )

        def assign_height(f):

            stats = open_buildings.reduceRegion(
                reducer=ee.Reducer.median(),
                geometry=f.geometry(),
                scale=4,
                bestEffort=True,
                maxPixels=1e10
            )

            h = ee.Number(
                ee.Algorithms.If(
                    stats.contains("building_height"),
                    stats.get("building_height"),
                    0
                )
            )

            floors = h.divide(3).ceil()

            return (
                f.set("height_m", h)
                 .set("floors", floors)
            )

        buildings_h = buildings.map(assign_height)

        roads = (
            ee.FeatureCollection(
                "projects/sat-io/open-datasets/GRIP4/South-East-Asia"
            )
            .filterBounds(roi)
        )

        roads_poly = roads.map(
            lambda f: f.buffer(7)
        )

        s2 = (
            ee.ImageCollection(
                "COPERNICUS/S2_SR_HARMONIZED"
            )
            .filterBounds(roi)
            .filterDate("2024-01-01", "2024-12-31")
            .filter(
                ee.Filter.lt(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    20
                )
            )
            .select(["B4", "B8"])
            .median()
            .clip(roi)
        )

        ndvi = s2.normalizedDifference(["B8", "B4"])

        vegetation_mask = ndvi.gt(0.35).unmask(0)

        def area_image(mask):

            stats = (
                mask.multiply(ee.Image.pixelArea())
                .rename("a")
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=roi,
                    scale=10,
                    bestEffort=True,
                    maxPixels=1e10
                )
            )

            return ee.Number(
                ee.Algorithms.If(
                    stats.contains("a"),
                    stats.get("a"),
                    0
                )
            ).divide(1e4)

        building_mask = (
            buildings_h
            .map(lambda f: f.set("m", 1))
            .reduceToImage(["m"], ee.Reducer.max())
            .unmask(0)
        )

        road_mask = (
            roads_poly
            .map(lambda f: f.set("m", 1))
            .reduceToImage(["m"], ee.Reducer.max())
            .unmask(0)
        )

        building_area = (
            buildings.geometry()
            .area()
            .divide(1e4)
        )

        roads_area = (
            roads_poly.geometry()
            .area()
            .divide(1e4)
        )

        vegetation_area = area_image(
            vegetation_mask
        )

        total_area = roi.area().divide(1e4)

        occupied = (
            building_mask
            .Or(road_mask)
            .Or(vegetation_mask)
        )

        open_mask = ee.Image.constant(1).clip(roi)
        open_mask = open_mask.updateMask(
            occupied.Not()
        )

        open_area = area_image(
            open_mask.unmask(0)
        )

        avg_height = ee.Number(
            ee.Algorithms.If(
                buildings_h.aggregate_mean("height_m"),
                buildings_h.aggregate_mean("height_m"),
                0
            )
        )

        avg_floors = ee.Number(
            ee.Algorithms.If(
                buildings_h.aggregate_mean("floors"),
                buildings_h.aggregate_mean("floors"),
                0
            )
        )

        builtup_proxy = (
            building_area.multiply(avg_floors)
        )

        far_proxy = (
            builtup_proxy.divide(total_area)
        )

        results = {
            "total": float(total_area.getInfo()),
            "building": float(building_area.getInfo()),
            "roads": float(roads_area.getInfo()),
            "veg": float(vegetation_area.getInfo()),
            "open": float(open_area.getInfo()),
            "avg_height": float(avg_height.getInfo()),
            "avg_floors": float(avg_floors.getInfo()),
            "builtup": float(builtup_proxy.getInfo()),
            "far": float(far_proxy.getInfo())
        }

        st.header("Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric("Buildings (ha)", round(results["building"],2))
        c2.metric("Vegetation (ha)", round(results["veg"],2))
        c3.metric("Open Space (ha)", round(results["open"],2))

        st.write(results)

        labels = [
            "Buildings",
            "Roads",
            "Vegetation",
            "Open",
            "Built-up"
        ]

        values = [
            results["building"],
            results["roads"],
            results["veg"],
            results["open"],
            results["builtup"]
        ]

        angles = np.linspace(
            0,
            2*np.pi,
            len(values),
            endpoint=False
        )

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(
            111,
            polar=True
        )

        ax.bar(
            angles,
            values,
            width=0.6
        )

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        st.pyplot(fig)

        st.header("Interactive Map")

        m = folium.Map(
            location=[lat, lon],
            zoom_start=15
        )

        m.add_ee_layer(
            building_mask.selfMask(),
            {"palette":["red"]},
            "Buildings"
        )

        m.add_ee_layer(
            road_mask.selfMask(),
            {"palette":["gray"]},
            "Roads"
        )

        m.add_ee_layer(
            vegetation_mask.selfMask(),
            {"palette":["green"]},
            "Vegetation"
        )

        m.add_ee_layer(
            open_mask.selfMask(),
            {"palette":["yellow"]},
            "Open Space"
        )

        folium.LayerControl().add_to(m)

        st_folium(
            m,
            height=600,
            width=1000
        )
