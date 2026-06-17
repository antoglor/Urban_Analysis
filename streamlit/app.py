import streamlit as st
from streamlit_folium import st_folium
import folium
import ee
import matplotlib.pyplot as plt
import numpy as np
from google.oauth2 import service_account

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Urban Analysis Tool",
    layout="wide"
)

st.title("Urban Analysis Tool")

# =====================================================
# EARTH ENGINE AUTH
# =====================================================

sa = st.secrets["google_service_account"]

credentials = service_account.Credentials.from_service_account_info(
    dict(sa),
    scopes=["https://www.googleapis.com/auth/earthengine"]
)

try:
    ee.Initialize(credentials)
except Exception:
    ee.Initialize()

# =====================================================
# FOLIUM + EE HELPER
# =====================================================

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

# =====================================================
# SESSION STATE
# =====================================================

if "clicked_lat" not in st.session_state:
    st.session_state.clicked_lat = None

if "clicked_lon" not in st.session_state:
    st.session_state.clicked_lon = None

if "results" not in st.session_state:
    st.session_state.results = None

# =====================================================
# ANALYSIS FUNCTION
# =====================================================

def run_analysis(lat, lon):

    roi = ee.Geometry.Point([lon, lat]).buffer(1000)

    # -------------------------------------------------
    # BUILDINGS
    # -------------------------------------------------

    buildings = (
        ee.FeatureCollection(
            "projects/sat-io/open-datasets/MSBuildings/India"
        )
        .filterBounds(roi)
        .filter(ee.Filter.gte("area", 16))
    )

    # -------------------------------------------------
    # OPEN BUILDINGS HEIGHT
    # -------------------------------------------------

    height_img = (
        ee.ImageCollection(
            "GOOGLE/Research/open-buildings-temporal/v1"
        )
        .filterBounds(roi)
        .select("building_height")
        .median()
        .clip(roi)
    )

    def assign_height(feature):

        stats = height_img.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=feature.geometry(),
            scale=4,
            bestEffort=True,
            maxPixels=1e10
        )

        height = ee.Number(
            ee.Algorithms.If(
                stats.contains("building_height"),
                stats.get("building_height"),
                0
            )
        )

        floors = height.divide(3).ceil()

        return (
            feature
            .set("height_m", height)
            .set("floors", floors)
        )

    buildings_h = buildings.map(assign_height)

    # -------------------------------------------------
    # ROADS
    # -------------------------------------------------

    roads = (
        ee.FeatureCollection(
            "projects/sat-io/open-datasets/GRIP4/South-East-Asia"
        )
        .filterBounds(roi)
    )

    roads_poly = roads.map(
        lambda f: f.buffer(7)
    )

    # -------------------------------------------------
    # VEGETATION
    # -------------------------------------------------

    s2 = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(roi)
        .filterDate(
            "2024-01-01",
            "2024-12-31"
        )
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

    ndvi = s2.normalizedDifference(
        ["B8", "B4"]
    )

    vegetation_mask = ndvi.gt(0.35).unmask(0)

    # -------------------------------------------------
    # MASKS
    # -------------------------------------------------

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

    occupied = (
        building_mask
        .Or(road_mask)
        .Or(vegetation_mask)
    )

    open_mask = (
        ee.Image.constant(1)
        .clip(roi)
        .updateMask(occupied.Not())
    )

    # -------------------------------------------------
    # AREA HELPER
    # -------------------------------------------------

    def raster_area(mask):

        stats = (
            mask
            .multiply(ee.Image.pixelArea())
            .rename("area")
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
                stats.contains("area"),
                stats.get("area"),
                0
            )
        ).divide(1e4)

    # -------------------------------------------------
    # AREAS
    # -------------------------------------------------

    total_area = roi.area().divide(1e4)

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

    vegetation_area = raster_area(
        vegetation_mask
    )

    open_area = raster_area(
        open_mask.unmask(0)
    )

    # -------------------------------------------------
    # BUILDING METRICS
    # -------------------------------------------------

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

    return {
        "lat": lat,
        "lon": lon,
        "total": float(total_area.getInfo()),
        "building": float(building_area.getInfo()),
        "roads": float(roads_area.getInfo()),
        "veg": float(vegetation_area.getInfo()),
        "open": float(open_area.getInfo()),
        "avg_height": float(avg_height.getInfo()),
        "avg_floors": float(avg_floors.getInfo()),
        "builtup": float(builtup_proxy.getInfo()),
        "far": float(far_proxy.getInfo()),
        "building_mask": building_mask.selfMask(),
        "road_mask": road_mask.selfMask(),
        "veg_mask": vegetation_mask.selfMask(),
        "open_mask": open_mask.selfMask()
    }

# =====================================================
# LOCATION PICKER
# =====================================================

st.subheader("Select Location")

m = folium.Map(
    location=[28.6, 77.2],
    zoom_start=5,
    tiles="CartoDB positron"
)

if st.session_state.clicked_lat is not None:

    folium.Circle(
        location=[
            st.session_state.clicked_lat,
            st.session_state.clicked_lon
        ],
        radius=500,
        color="blue",
        fill=True
    ).add_to(m)

map_data = st_folium(
    m,
    height=350,
    width=900
)

if map_data and map_data.get("last_clicked"):

    st.session_state.clicked_lat = (
        map_data["last_clicked"]["lat"]
    )

    st.session_state.clicked_lon = (
        map_data["last_clicked"]["lng"]
    )

st.write(
    "Selected:",
    st.session_state.clicked_lat,
    st.session_state.clicked_lon
)

# =====================================================
# RUN
# =====================================================

if st.button("Run Analysis"):

    if st.session_state.clicked_lat is None:

        st.error(
            "Please select a location first."
        )

    else:

        with st.spinner("Running analysis..."):

            st.session_state.results = run_analysis(
                st.session_state.clicked_lat,
                st.session_state.clicked_lon
            )

# =====================================================
# RESULTS
# =====================================================

if st.session_state.results:

    r = st.session_state.results

    st.header("Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Buildings (ha)",
        round(r["building"], 2)
    )

    c2.metric(
        "Roads (ha)",
        round(r["roads"], 2)
    )

    c3.metric(
        "Vegetation (ha)",
        round(r["veg"], 2)
    )

    c4.metric(
        "Open Space (ha)",
        round(r["open"], 2)
    )

    st.write(
        f"Average Height: {r['avg_height']:.2f} m"
    )

    st.write(
        f"Average Floors: {r['avg_floors']:.2f}"
    )

    st.write(
        f"Built-up Proxy: {r['builtup']:.2f} ha"
    )

    st.write(
        f"FAR Proxy: {r['far']:.2f}"
    )

    # -------------------------------------------------
    # POLAR CHART
    # -------------------------------------------------

    labels = [
        "Buildings",
        "Roads",
        "Vegetation",
        "Open",
        "Built-up"
    ]

    values = [
        r["building"],
        r["roads"],
        r["veg"],
        r["open"],
        r["builtup"]
    ]

    angles = np.linspace(
        0,
        2*np.pi,
        len(values),
        endpoint=False
    )

    fig = plt.figure(figsize=(5, 5))
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

    # -------------------------------------------------
    # RESULT MAP
    # -------------------------------------------------

    st.subheader("Map")

    rm = folium.Map(
        location=[r["lat"], r["lon"]],
        zoom_start=15
    )

    rm.add_ee_layer(
        r["building_mask"],
        {"palette": ["red"]},
        "Buildings"
    )

    rm.add_ee_layer(
        r["road_mask"],
        {"palette": ["gray"]},
        "Roads"
    )

    rm.add_ee_layer(
        r["veg_mask"],
        {"palette": ["green"]},
        "Vegetation"
    )

    rm.add_ee_layer(
        r["open_mask"],
        {"palette": ["yellow"]},
        "Open"
    )

    folium.LayerControl().add_to(rm)

    st_folium(
        rm,
        height=600,
        width=1000
    )
