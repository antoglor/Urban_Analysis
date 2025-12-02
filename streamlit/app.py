# app.py (cleaned)
import streamlit as st
from streamlit_folium import st_folium
import ee
import geemap.foliumap as geemap
import folium

import streamlit as st
from streamlit_folium import st_folium
import ee
import geemap.foliumap as geemap
import folium
from google.oauth2 import service_account

service_account_info = json.loads(st.secrets["google_service_account_json"]["value"])
credentials = service_account.Credentials.from_service_account_info(service_account_info)
ee.Initialize(credentials=credentials)


st.set_page_config(layout="wide")
st.title("Urban Analysis Tool — Compare built up area for any selected latitude longitude")
st.write("Click to select location, then Run Analysis")

# Session state for clicked coords
if "clicked_lat" not in st.session_state:
    st.session_state.clicked_lat = None
if "clicked_lon" not in st.session_state:
    st.session_state.clicked_lon = None

# ---------- Selector map ----------
st.markdown("### Select location- map will zoom out after selection")
st.write("Click on the small map to pick a point (lat/lon). A preview circle will appear.")

# Default center
default_center = [28.6596, 77.1295]

# Create the base map
selector_map = geemap.Map(center=default_center, zoom=5)

# If the user already clicked earlier, draw the circle
if "clicked_lat" in st.session_state and st.session_state.clicked_lat is not None:
    lat = st.session_state.clicked_lat
    lon = st.session_state.clicked_lon
    
    folium.Circle(
        radius=500,                      # preview radius (meters)
        location=(lat, lon),
        color="blue",
        fill=True,
        fill_opacity=0.15,               # faint circle
        weight=1
    ).add_to(selector_map)

# Read map clicks
map_data = st_folium(selector_map, height=320, width=700)

# Capture clicks
if map_data and map_data.get("last_clicked"):
    st.session_state.clicked_lat = map_data["last_clicked"]["lat"]
    st.session_state.clicked_lon = map_data["last_clicked"]["lng"]
    st.success(
        f"Selected: {st.session_state.clicked_lat:.6f}, {st.session_state.clicked_lon:.6f}"
    )

# Display lat/lon readout
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.write("**Latitude**")
    st.write(st.session_state.get("clicked_lat", "—"))
with col2:
    st.write("**Longitude**")
    st.write(st.session_state.get("clicked_lon", "—"))
with col3:
    st.write("")


# ---------- Run Analysis ----------
if st.button("Run Analysis"):
    if st.session_state.clicked_lat is None:
        st.error("Please click the map first to select a location.")
        st.stop()

    lat = st.session_state.clicked_lat
    lon = st.session_state.clicked_lon
    st.write("Running analysis for:", lat, lon)

    try:
        with st.spinner("Running Earth Engine processing (may take a few minutes)..."):
            # -------------------------
            # SETUP ROI and Data
            # -------------------------
            roi = ee.Geometry.Point([lon, lat]).buffer(1000)  # 1 km radius

            # BUILDINGS (MSBuildings - adjust collection as needed)
            buildings = (
                ee.FeatureCollection("projects/sat-io/open-datasets/MSBuildings/India")
                .filterBounds(roi)
            ).filter(ee.Filter.gte('area', 16))

            # BUILDING HEIGHT (Open Buildings)
            open_buildings = (
                ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1")
                .filterBounds(roi).select("building_height")
            ).median().clip(roi)

            # Assign median building height to each building feature (safe If)
            def assign_height(feature):
                stats = open_buildings.reduceRegion(
                    reducer=ee.Reducer.median(),
                    geometry=feature.geometry(),
                    scale=4,
                    maxPixels=1e10,
                    bestEffort=True
                )
                hv = ee.Algorithms.If(stats.contains("building_height"), stats.get("building_height"), 0)
                return feature.set("height_m", ee.Number(hv))

            buildings_height = buildings.map(assign_height)

            # Compute floors, area, volume
            def compute_floors_volume(feature):
                height_m = ee.Number(feature.get("height_m"))
                floors = height_m.divide(3).ceil()
                area_m2 = feature.geometry().area()
                volume_m3 = height_m.multiply(area_m2)
                return feature.set("floors", floors).set("volume_m3", volume_m3).set("area_m2", area_m2)

            buildings_height = buildings_height.map(compute_floors_volume)

            # ROADS (GRIP4) - note: you used SEA dataset; filterBounds(roi) will limit features
            roads = ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/South-East-Asia").filterBounds(roi)
            roads_poly = roads.map(lambda f: f.buffer(7))

            # VEGETATION (Sentinel-2 NDVI)
            s2 = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(roi)
                .filterDate("2024-01-01", "2024-12-31")
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                .select(["B4", "B8"])
            ).median().clip(roi)
            ndvi = s2.normalizedDifference(["B8", "B4"])

            # Create a single, explicit binary vegetation mask (1 for veg, 0 otherwise)
            ndvi_thresh = 0.35
            vegetation_mask = ndvi.gt(ndvi_thresh).rename("veg").unmask(0)  # 1/0 image

            # -------------------------
            # AREA HELPERS (robust)
            # -------------------------
            def area_featurecollection_hectares(fc):
                # Area of the union of features (hectares)
                return fc.geometry().area().divide(1e4)

            def area_image_mask_hectares(mask_image, region, scale=10):
                pixel_area = ee.Image.pixelArea()
                area_image = mask_image.multiply(pixel_area).rename("area")
                stats = area_image.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=region,
                    scale=scale,
                    maxPixels=1e10,
                    bestEffort=True,
                    tileScale=4
                )
                # safe return of 0 if 'area' missing
                m2 = ee.Number(ee.Algorithms.If(stats.contains("area"), stats.get("area"), 0))
                return m2.divide(1e4)

            # -------------------------
            # LAND-USE AREAS (server-side)
            # -------------------------
            total_area = roi.area().divide(1e4)   # EE Number (ha)
            building_area = area_featurecollection_hectares(buildings)
            roads_area = area_featurecollection_hectares(roads_poly)
            vegetation_area = area_image_mask_hectares(vegetation_mask, roi, scale=10)

            # -------------------------
            # OPEN SPACE (vectorize from mask) - canonical open space
            # -------------------------
            base = ee.Image.constant(1).clip(roi)
            buildings_mask_img = buildings_height.map(lambda f: f.set('mask', 1)) \
                .reduceToImage(['mask'], ee.Reducer.max()).unmask(0)
            roads_mask_img = roads_poly.map(lambda f: f.set('mask', 1)) \
                .reduceToImage(['mask'], ee.Reducer.max()).unmask(0)
            veg_mask_img = vegetation_mask  # 1/0

            occupied = buildings_mask_img.Or(roads_mask_img).Or(veg_mask_img)
            open_space_img = base.updateMask(occupied.Not())

            open_fc_raw = open_space_img.reduceToVectors(
                geometry=roi, scale=10, geometryType='polygon',
                eightConnected=True, labelProperty='pixel_count', maxPixels=1e10
            )

            open_fc_raw = open_fc_raw.map(lambda f: f.set("area_m2", ee.Number(f.get("pixel_count")).multiply(100)))
            buffer_dist = 15
            open_fc = open_fc_raw.map(lambda f: f.buffer(buffer_dist).buffer(-buffer_dist))
            open_fc = open_fc.map(lambda f: f.set("area_m2", f.geometry().area()))

            open_large = open_fc.filter(ee.Filter.gte("area_m2", 1000))
            open_small = open_fc.filter(ee.Filter.lt("area_m2", 1000))

            open_large_ha = open_large.aggregate_sum("area_m2").divide(1e4)
            open_small_ha = open_small.aggregate_sum("area_m2").divide(1e4)
            total_open_space_ha = ee.Number(open_large_ha).add(ee.Number(open_small_ha))

            # -------------------------
            # AGGREGATES / PERCENTAGES (keep server-side)
            # -------------------------
            # safeguard aggregates that might be null
            avg_height = buildings_height.aggregate_mean("height_m")
            avg_height = ee.Number(ee.Algorithms.If(avg_height, avg_height, 0))

            avg_floors = buildings_height.aggregate_mean("floors")
            avg_floors = ee.Number(ee.Algorithms.If(avg_floors, avg_floors, 0))

            # Percentages (EE Numbers)
            buildings_pct = building_area.divide(total_area).multiply(100)
            roads_pct = roads_area.divide(total_area).multiply(100)
            veg_pct = vegetation_area.divide(total_area).multiply(100)
            large_open_pct = open_large_ha.divide(total_area).multiply(100)
            total_open_pct = total_open_space_ha.divide(total_area).multiply(100)

            # Built-up area proxy and FAR: compute server-side to avoid mixing types
            total_built_up_area_proxy_ha = building_area.multiply(avg_floors)  # building_area in ha * floors -> ha * floors (proxy)
            # If you want proxy in m2: building_area.multiply(1e4).multiply(avg_floors)
            total_FAR_proxy = total_built_up_area_proxy_ha.divide(total_area)  # proxy FAR in unitless ratio

            # -------------------------
            # PACKAGE results (convert to Python values with getInfo())
            # -------------------------
            results = {}
            results["total_ha"] = float(total_area.getInfo())
            results["building_ha"] = float(building_area.getInfo())
            results["roads_ha"] = float(roads_area.getInfo())
            results["veg_ha"] = float(vegetation_area.getInfo())
            results["open_ha"] = float(total_open_space_ha.getInfo())
            results["large_open_ha"] = float(open_large_ha.getInfo()) if open_large_ha else 0.0
            results["small_open_ha"] = float(open_small_ha.getInfo()) if open_small_ha else 0.0

            results["buildings_pct"] = float(buildings_pct.getInfo())
            results["roads_pct"] = float(roads_pct.getInfo())
            results["veg_pct"] = float(veg_pct.getInfo())
            results["large_open_pct"] = float(large_open_pct.getInfo())
            results["total_open_pct"] = float(total_open_pct.getInfo())

            results["avg_height"] = float(avg_height.getInfo())
            results["avg_floors"] = float(avg_floors.getInfo())
            results["total_built_up_area_proxy_ha"] = float(total_built_up_area_proxy_ha.getInfo())
            results["total_FAR_proxy"] = float(total_FAR_proxy.getInfo())

            # Keep EE images for visualization (no getInfo)
            buildings_mask_viz = buildings_mask_img.gt(0).selfMask()
            roads_mask_viz = roads_mask_img.gt(0).selfMask()
            veg_mask_viz = veg_mask_img.gt(0).selfMask()
            open_mask_viz = open_space_img.selfMask()

            results["roi"] = roi
            results["open_mask_img"] = open_mask_viz
            results["buildings_mask_img"] = buildings_mask_viz
            results["roads_mask_img"] = roads_mask_viz
            results["veg_mask_img"] = veg_mask_viz

    except Exception as e:
        st.error(f"Earth Engine error: {e}")
        results = None

    # -------------------------
    # Display results & map
    # -------------------------
    if results:
        st.header("Land Use Summary (hectares)")
        st.write("Total AOI (ha):", results["total_ha"])
        st.write("Buildings (ha):", results["building_ha"])
        st.write("Roads (ha):", results["roads_ha"])
        st.write("Vegetation (ha):", results["veg_ha"])
        st.write("Open space (ha):", results["open_ha"])
        st.write("Large open areas (ha):", results["large_open_ha"])
        st.write("Small open areas (ha):", results["small_open_ha"])

        st.header("Percent share")
        st.write("Buildings %:", results["buildings_pct"])
        st.write("Roads %:", results["roads_pct"])
        st.write("Vegetation %:", results["veg_pct"])
        #st.write("Large open areas %:", results["large_open_pct"])
        st.write("Total open %:", results["total_open_pct"])

        st.header("Building metrics")
        st.write("Average building height (m):", results["avg_height"])
        st.write("Average number of floors:", results["avg_floors"])
        st.write("Total building area proxy (ha):", results["total_built_up_area_proxy_ha"])
        st.write("Total building FAR (proxy):", results["total_FAR_proxy"])

                
        
        
        # ---------------------------------------------------------
        #  CIRCULAR CHARTS (replace the entire Result map section)
        # ---------------------------------------------------------
        import matplotlib.pyplot as plt
        import numpy as np
        
        st.header("Circular Bar Chart — Land Use Distribution")
        
        # ----------------------
        # LAND-USE PLOT (compact)
        # ----------------------
        lu_labels = [
            "Buildings (ha)",
            "Roads (ha)",
            "Vegetation (ha)",
            "Open space (ha)",
            "Total Built Up Area"
        ]
        
        lu_values = [
            results["building_ha"],
            results["roads_ha"],
            results["veg_ha"],
            results["open_ha"],
            results["total_built_up_area_proxy_ha"]
        ]
        
        # convert to radians
        angles = np.linspace(0, 2 * np.pi, len(lu_values), endpoint=False).tolist()
        
        # blue/green monochrome palette (5 colors now)
        lu_colors = ["#0D47A1", "#1976D2", "#26A69A", "#4DB6AC", "#80CBC4"]
        
        # SMALLER FIGURE SIZE
        fig1 = plt.figure(figsize=(5, 5))  # was 7x7
        ax1 = fig1.add_subplot(111, polar=True)
        
        bars = ax1.bar(
            x=angles,
            height=lu_values,
            width=0.6,          # slimmer bars (was 0.8)
            bottom=0,
            color=lu_colors,
            alpha=0.85
        )
        
        for angle, height, label in zip(angles, lu_values, lu_labels):
            ax1.text(
                angle,
                height + max(lu_values) * 0.03,  # slightly tighter spacing
                f"{label}\n{height:.1f} ha",
                ha="center",
                va="center",
                fontsize=8,        # smaller font (was 10)
                rotation=np.degrees(angle),
                rotation_mode="anchor",
                color="#0A2E36"
            )
        
        # styling
        ax1.set_theta_offset(np.pi / 2)
        ax1.set_theta_direction(-1)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        
        colA, colB = st.columns(2)
        with colA:
            st.pyplot(fig1)


        
        
       



else:
    st.info("Click 'Run Analysis' after selecting a location.")
