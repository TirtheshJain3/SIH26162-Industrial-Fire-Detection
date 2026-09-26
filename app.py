import streamlit as st
import pandas as pd
import numpy as np
import folium
import joblib
import re

from folium.plugins import HeatMap
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SIH26162 | Industrial Fire Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 1.5rem;
}

h1 {
    color: #ff4b4b;
}

h2, h3 {
    color: #ffffff;
}

.metric-card {
    background: linear-gradient(135deg, #161b22, #21262d);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #30363d;
    text-align: center;
}

.metric-title {
    color: #8b949e;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 30px;
    font-weight: bold;
}

.alert-high {
    background-color: #3d1111;
    border-left: 5px solid #ff3333;
    padding: 15px;
    border-radius: 8px;
}

.alert-medium {
    background-color: #3d2c0a;
    border-left: 5px solid #ffa500;
    padding: 15px;
    border-radius: 8px;
}

.alert-low {
    background-color: #123d1c;
    border-left: 5px solid #00cc66;
    padding: 15px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE LOADING
# ============================================================

@st.cache_data
def load_prediction_data():

    df = pd.read_csv(
        "SIH26162_Prediction_Data.csv"
    )

    return df


@st.cache_data
def load_osm_data():

    try:
        df = pd.read_csv(
            "SIH26162_OSM_Industrial_Data.csv"
        )
        return df

    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_model():

    try:
        return joblib.load(
            "SIH26162_RandomForest_Model.pkl"
        )

    except Exception:
        return None


@st.cache_resource
def load_feature_list():

    try:
        features = joblib.load(
            "SIH26162_Feature_List.pkl"
        )

        return list(features)

    except Exception:
        return []


# ============================================================
# LOAD DATA
# ============================================================

try:

    prediction_data = load_prediction_data()
    osm_data = load_osm_data()
    rf_model = load_model()
    feature_columns = load_feature_list()

except Exception as e:

    st.error("❌ Unable to load application data.")

    st.code(str(e))

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):

    if df is None or df.empty:
        return None

    lower_columns = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in possible_names:

        if name.lower() in lower_columns:
            return lower_columns[name.lower()]

    return None


def get_lat_lon(df):

    lat_col = find_column(
        df,
        [
            "latitude",
            "lat",
            "Latitude",
            "LATITUDE"
        ]
    )

    lon_col = find_column(
        df,
        [
            "longitude",
            "lon",
            "lng",
            "Longitude",
            "LONGITUDE"
        ]
    )

    return lat_col, lon_col


def get_frp_column(df):

    return find_column(
        df,
        [
            "frp",
            "FRP",
            "frp_mw",
            "FRP_MW"
        ]
    )


def get_classification_column(df):

    return find_column(
        df,
        [
            "classification",
            "class",
            "label",
            "fire_class",
            "predicted_class"
        ]
    )


def get_risk_column(df):

    return find_column(
        df,
        [
            "risk_level",
            "risk",
            "risk_category",
            "Risk_Level"
        ]
    )


def get_score_column(df):

    return find_column(
        df,
        [
            "risk_score",
            "score",
            "Risk_Score"
        ]
    )


def safe_numeric(series, default=0):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(default)


def normalize_intensity(series):

    values = safe_numeric(series, 1)

    if len(values) == 0:
        return values

    max_value = values.max()

    if max_value <= 0:
        return pd.Series(
            np.ones(len(values)),
            index=values.index
        )

    return values / max_value


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔥 SIH26162")

st.sidebar.markdown(
    "**Industrial Fire Intelligence System**"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🗺️ Fire Risk Map",
        "🔥 Fire Detections",
        "📊 Analytics",
        "🚨 Alerts",
        "🤖 AI Model"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "NASA FIRMS • OSM • GIS • Machine Learning"
)


# ============================================================
# COMMON DATA INFORMATION
# ============================================================

lat_col, lon_col = get_lat_lon(prediction_data)

frp_col = get_frp_column(prediction_data)

class_col = get_classification_column(prediction_data)

risk_col = get_risk_column(prediction_data)

score_col = get_score_column(prediction_data)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🔥 Industrial Fire Intelligence System")

    st.markdown(
        """
        ### AI-Based Detection and Classification of
        Industrial Fires and Persistent Thermal Sources

        **SIH26162**
        """
    )

    st.markdown("---")

    total_detections = len(prediction_data)

    if class_col:

        classes = (
            prediction_data[class_col]
            .astype(str)
            .str.lower()
        )

        fire_count = classes.str.contains(
            "fire|industrial fire",
            regex=True
        ).sum()

        thermal_count = total_detections - fire_count

    else:

        fire_count = 0
        thermal_count = total_detections

    if risk_col:

        risk_values = (
            prediction_data[risk_col]
            .astype(str)
            .str.lower()
        )

        high_risk = risk_values.str.contains(
            "high|critical"
        ).sum()

    else:

        high_risk = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-title">
            THERMAL DETECTIONS
            </div>
            <div class="metric-value">
            {total_detections}
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-title">
            FIRE CANDIDATES
            </div>
            <div class="metric-value">
            {fire_count}
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-title">
            HIGH RISK
            </div>
            <div class="metric-value">
            {high_risk}
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
            <div class="metric-title">
            INDUSTRIAL LOCATIONS
            </div>
            <div class="metric-value">
            {len(osm_data)}
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("📌 System Pipeline")

    st.markdown(
        """
        **NASA FIRMS**
        → Thermal Hotspots
        → **OSM Industrial Context**
        → Feature Engineering
        → Persistence Analysis
        → **Random Forest**
        → Fire Classification
        → Risk Scoring
        → **Interactive GIS Map**
        """
    )

    st.markdown("---")

    st.subheader("📊 Dataset Preview")

    st.dataframe(
        prediction_data.head(20),
        use_container_width=True
    )


# ============================================================
# FIRE RISK MAP
# ============================================================

elif page == "🗺️ Fire Risk Map":

    st.title("🗺️ Industrial Fire & Thermal Risk Map")

    st.markdown(
        """
        The map combines **NASA thermal detections**, 
        **FRP-based heat intensity**, and **industrial locations**.
        """
    )

    if lat_col is None or lon_col is None:

        st.error(
            "❌ Latitude and longitude columns were not found."
        )

        st.write(
            "Available columns:"
        )

        st.write(
            prediction_data.columns.tolist()
        )

        st.stop()

    # --------------------------------------------------------
    # PREPARE MAP DATA
    # --------------------------------------------------------

    map_data = prediction_data.copy()

    map_data[lat_col] = pd.to_numeric(
        map_data[lat_col],
        errors="coerce"
    )

    map_data[lon_col] = pd.to_numeric(
        map_data[lon_col],
        errors="coerce"
    )

    map_data = map_data.dropna(
        subset=[lat_col, lon_col]
    )

    map_data = map_data[
        (map_data[lat_col] >= -90) &
        (map_data[lat_col] <= 90) &
        (map_data[lon_col] >= -180) &
        (map_data[lon_col] <= 180)
    ]

    if len(map_data) == 0:

        st.error(
            "❌ No valid geographic coordinates found."
        )

        st.stop()

    # --------------------------------------------------------
    # MAP CENTER
    # --------------------------------------------------------

    center_lat = map_data[lat_col].mean()
    center_lon = map_data[lon_col].mean()

    # --------------------------------------------------------
    # CREATE MAP
    # --------------------------------------------------------

    m = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=7,
        control_scale=True,
        tiles=None
    )

    # --------------------------------------------------------
    # SATELLITE BASEMAP
    # --------------------------------------------------------

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="🛰️ Satellite",
        overlay=False,
        control=True
    ).add_to(m)

    # --------------------------------------------------------
    # STREET MAP
    # --------------------------------------------------------

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="🗺️ Street Map",
        overlay=False,
        control=True
    ).add_to(m)

    # ========================================================
    # 🔥 THERMAL HEATMAP
    # ========================================================

    heat_data = []

    if frp_col:

        map_data[frp_col] = pd.to_numeric(
            map_data[frp_col],
            errors="coerce"
        )

        map_data[frp_col] = map_data[
            frp_col
        ].fillna(1)

        max_frp = map_data[
            frp_col
        ].max()

        if max_frp > 0:

            map_data["_heat_intensity"] = (
                map_data[frp_col] /
                max_frp
            )

        else:

            map_data["_heat_intensity"] = 1.0

    else:

        map_data["_heat_intensity"] = 1.0

    for _, row in map_data.iterrows():

        intensity = float(
            row["_heat_intensity"]
        )

        intensity = max(
            0.1,
            min(1.0, intensity)
        )

        heat_data.append(
            [
                float(row[lat_col]),
                float(row[lon_col]),
                intensity
            ]
        )

    # --------------------------------------------------------
    # ADD HEATMAP
    # --------------------------------------------------------

    if len(heat_data) > 0:

        HeatMap(
            heat_data,
            name="🔥 Thermal Intensity Heatmap",
            radius=28,
            blur=22,
            min_opacity=0.35,
            max_zoom=10,
            gradient={
                0.15: "blue",
                0.35: "cyan",
                0.55: "lime",
                0.70: "yellow",
                0.85: "orange",
                1.00: "red"
            }
        ).add_to(m)

    # ========================================================
    # INDUSTRIAL LOCATIONS
    # ========================================================

    industrial_group = folium.FeatureGroup(
        name="🏭 Industrial Locations",
        show=True
    )

    if not osm_data.empty:

        osm_lat_col, osm_lon_col = get_lat_lon(
            osm_data
        )

        if (
            osm_lat_col is not None and
            osm_lon_col is not None
        ):

            osm_copy = osm_data.copy()

            osm_copy[osm_lat_col] = pd.to_numeric(
                osm_copy[osm_lat_col],
                errors="coerce"
            )

            osm_copy[osm_lon_col] = pd.to_numeric(
                osm_copy[osm_lon_col],
                errors="coerce"
            )

            osm_copy = osm_copy.dropna(
                subset=[
                    osm_lat_col,
                    osm_lon_col
                ]
            )

            for _, row in osm_copy.iterrows():

                lat = float(
                    row[osm_lat_col]
                )

                lon = float(
                    row[osm_lon_col]
                )

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    color="blue",
                    fill=True,
                    fill_color="blue",
                    fill_opacity=0.85,
                    popup=folium.Popup(
                        "<b>🏭 Industrial Location</b>",
                        max_width=250
                    )
                ).add_to(
                    industrial_group
                )

    industrial_group.add_to(m)

    # ========================================================
    # THERMAL DETECTION MARKERS
    # ========================================================

    detection_group = folium.FeatureGroup(
        name="🔥 Thermal Detections",
        show=True
    )

    for _, row in map_data.iterrows():

        lat = float(row[lat_col])
        lon = float(row[lon_col])

        if class_col:

            classification = str(
                row[class_col]
            )

        else:

            classification = "Thermal Detection"

        if risk_col:

            risk = str(
                row[risk_col]
            )

        else:

            risk = "Unknown"

        if frp_col:

            frp_value = row[frp_col]

        else:

            frp_value = "N/A"

        # ----------------------------------------------------
        # Marker color
        # ----------------------------------------------------

        text = (
            classification +
            " " +
            risk
        ).lower()

        if (
            "high" in text or
            "critical" in text or
            "fire" in text
        ):

            marker_color = "red"

        elif (
            "medium" in text or
            "moderate" in text
        ):

            marker_color = "orange"

        else:

            marker_color = "green"

        popup_html = f"""
        <div style="width:250px">

        <h4>🔥 Thermal Detection</h4>

        <b>Classification:</b>
        {classification}<br><br>

        <b>Risk:</b>
        {risk}<br><br>

        <b>FRP:</b>
        {frp_value}<br><br>

        <b>Latitude:</b>
        {lat:.5f}<br>

        <b>Longitude:</b>
        {lon:.5f}

        </div>
        """

        folium.CircleMarker(
            location=[
                lat,
                lon
            ],
            radius=6,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.9,
            popup=folium.Popup(
                popup_html,
                max_width=300
            )
        ).add_to(
            detection_group
        )

    detection_group.add_to(m)

    # ========================================================
    # LAYER CONTROL
    # ========================================================

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    # ========================================================
    # DISPLAY MAP
    # ========================================================

    st_folium(
        m,
        width=None,
        height=650,
        returned_objects=[]
    )

    st.success(
        f"🔥 Thermal heatmap loaded using "
        f"{len(heat_data)} detections."
    )


# ============================================================
# FIRE DETECTIONS
# ============================================================

elif page == "🔥 Fire Detections":

    st.title("🔥 Fire & Thermal Detections")

    data = prediction_data.copy()

    if class_col:

        selected_class = st.multiselect(
            "Filter Classification",
            sorted(
                data[class_col]
                .dropna()
                .astype(str)
                .unique()
            )
        )

        if selected_class:

            data = data[
                data[class_col]
                .astype(str)
                .isin(selected_class)
            ]

    if risk_col:

        selected_risk = st.multiselect(
            "Filter Risk Level",
            sorted(
                data[risk_col]
                .dropna()
                .astype(str)
                .unique()
            )
        )

        if selected_risk:

            data = data[
                data[risk_col]
                .astype(str)
                .isin(selected_risk)
            ]

    st.write(
        f"Showing **{len(data)}** detections"
    )

    st.dataframe(
        data,
        use_container_width=True,
        height=550
    )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Thermal Analytics")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "🔥 Classification Distribution"
        )

        if class_col:

            counts = (
                prediction_data[
                    class_col
                ]
                .astype(str)
                .value_counts()
            )

            st.bar_chart(counts)

        else:

            st.info(
                "Classification column not available."
            )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "🚨 Risk Distribution"
        )

        if risk_col:

            counts = (
                prediction_data[
                    risk_col
                ]
                .astype(str)
                .value_counts()
            )

            st.bar_chart(counts)

        else:

            st.info(
                "Risk column not available."
            )

    # --------------------------------------------------------
    # FRP
    # --------------------------------------------------------

    if frp_col:

        st.subheader(
            "🔥 Fire Radiative Power"
        )

        frp_values = pd.to_numeric(
            prediction_data[frp_col],
            errors="coerce"
        ).dropna()

        if len(frp_values) > 0:

            st.line_chart(
                frp_values.reset_index(
                    drop=True
                )
            )

            st.metric(
                "Maximum FRP",
                f"{frp_values.max():.2f}"
            )

            st.metric(
                "Average FRP",
                f"{frp_values.mean():.2f}"
            )


# ============================================================
# ALERTS
# ============================================================

elif page == "🚨 Alerts":

    st.title("🚨 Fire Risk Alerts")

    if risk_col:

        alert_data = prediction_data.copy()

        alert_data["_risk_text"] = (
            alert_data[risk_col]
            .astype(str)
            .str.lower()
        )

        high_alerts = alert_data[
            alert_data["_risk_text"].str.contains(
                "high|critical"
            )
        ]

        medium_alerts = alert_data[
            alert_data["_risk_text"].str.contains(
                "medium|moderate"
            )
        ]

        st.subheader(
            f"🔴 High / Critical Risk: {len(high_alerts)}"
        )

        if len(high_alerts) > 0:

            st.dataframe(
                high_alerts.drop(
                    columns=["_risk_text"],
                    errors="ignore"
                ),
                use_container_width=True
            )

        else:

            st.success(
                "No high-risk detections found."
            )

        st.subheader(
            f"🟠 Medium Risk: {len(medium_alerts)}"
        )

        if len(medium_alerts) > 0:

            st.dataframe(
                medium_alerts.drop(
                    columns=["_risk_text"],
                    errors="ignore"
                ),
                use_container_width=True
            )

    else:

        st.info(
            "Risk-level information is not available."
        )


# ============================================================
# AI MODEL
# ============================================================

elif page == "🤖 AI Model":

    st.title("🤖 AI Model")

    st.subheader(
        "Random Forest Classification"
    )

    if rf_model is not None:

        st.success(
            "✅ Random Forest model loaded successfully."
        )

        st.write(
            f"**Number of estimators:** "
            f"{getattr(rf_model, 'n_estimators', 'N/A')}"
        )

        st.write(
            f"**Number of features:** "
            f"{len(feature_columns)}"
        )

        if feature_columns:

            st.subheader(
                "📊 Model Features"
            )

            for feature in feature_columns:

                st.write(
                    f"• {feature}"
                )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        if hasattr(
            rf_model,
            "feature_importances_"
        ):

            importance_values = (
                rf_model.feature_importances_
            )

            n = min(
                len(feature_columns),
                len(importance_values)
            )

            importance_df = pd.DataFrame(
                {
                    "Feature":
                        feature_columns[:n],

                    "Importance":
                        importance_values[:n]
                }
            )

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
            )

            st.subheader(
                "📈 Feature Importance"
            )

            st.bar_chart(
                importance_df.set_index(
                    "Feature"
                )
            )

            st.dataframe(
                importance_df,
                use_container_width=True
            )

    else:

        st.error(
            "❌ Random Forest model could not be loaded."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SIH26162 | AI-Based Detection and Classification "
    "of Industrial Fires and Persistent Thermal Sources"
)

st.caption(
    "NASA FIRMS • OpenStreetMap • GIS • Random Forest • Streamlit"
)
