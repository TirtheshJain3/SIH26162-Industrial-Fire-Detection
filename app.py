import streamlit as st
import pandas as pd
import numpy as np
import folium
import joblib

from folium.plugins import HeatMap
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SIH26162 Industrial Fire Intelligence",
    page_icon="🔥",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.metric-card {
    background: linear-gradient(
        135deg,
        #161b22,
        #21262d
    );

    padding: 18px;

    border-radius: 12px;

    border: 1px solid #30363d;

    text-align: center;
}

.metric-title {
    color: #8b949e;
    font-size: 13px;
}

.metric-value {
    color: white;
    font-size: 30px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "SIH26162_Prediction_Data.csv"
    )


@st.cache_data
def load_industrial():

    try:

        return pd.read_csv(
            "SIH26162_OSM_Industrial_Data.csv"
        )

    except:

        return pd.DataFrame()


@st.cache_resource
def load_model():

    try:

        return joblib.load(
            "SIH26162_RandomForest_Model.pkl"
        )

    except:

        return None


@st.cache_resource
def load_features():

    try:

        return joblib.load(
            "SIH26162_Feature_List.pkl"
        )

    except:

        return []


prediction_data = load_data()

osm_data = load_industrial()

rf_model = load_model()

feature_columns = load_features()


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(df, names):

    for name in names:

        for col in df.columns:

            if str(col).lower() == name.lower():

                return col

    return None


lat_col = find_column(
    prediction_data,
    [
        "latitude",
        "lat"
    ]
)

lon_col = find_column(
    prediction_data,
    [
        "longitude",
        "lon",
        "lng"
    ]
)

frp_col = find_column(
    prediction_data,
    [
        "frp",
        "frp_mw"
    ]
)

classification_col = find_column(
    prediction_data,
    [
        "classification",
        "label"
    ]
)

risk_col = find_column(
    prediction_data,
    [
        "risk_level",
        "risk"
    ]
)

risk_score_col = find_column(
    prediction_data,
    [
        "risk_score",
        "score"
    ]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🔥 SIH26162"
)

st.sidebar.caption(
    "Industrial Fire Intelligence System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🗺️ Fire Risk Map",
        "🔥 Fire Classification",
        "📊 Analytics",
        "🚨 Alerts",
        "🤖 AI Model"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "🔥 Industrial Fire Intelligence System"
    )

    st.markdown(
        """
        ### AI-Based Detection, Classification and
        Monitoring of Industrial Fires and Persistent
        Thermal Sources
        """
    )

    st.markdown("---")

    total = len(
        prediction_data
    )

    if classification_col:

        industrial_fire = (
            prediction_data[
                classification_col
            ]
            .astype(str)
            .str.contains(
                "Industrial Fire",
                case=False,
                na=False
            )
            .sum()
        )

        persistent = (
            prediction_data[
                classification_col
            ]
            .astype(str)
            .str.contains(
                "Persistent",
                case=False,
                na=False
            )
            .sum()
        )

        natural = (
            prediction_data[
                classification_col
            ]
            .astype(str)
            .str.contains(
                "Natural",
                case=False,
                na=False
            )
            .sum()
        )

    else:

        industrial_fire = 0
        persistent = 0
        natural = 0

    if risk_col:

        high_risk = (
            prediction_data[
                risk_col
            ]
            .astype(str)
            .str.contains(
                "High|Critical",
                case=False,
                na=False
            )
            .sum()
        )

    else:

        high_risk = 0


    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "Thermal Detections",
            total
        )

    with c2:

        st.metric(
            "Industrial Fires",
            industrial_fire
        )

    with c3:

        st.metric(
            "Persistent Sources",
            persistent
        )

    with c4:

        st.metric(
            "Natural / Forest",
            natural
        )

    with c5:

        st.metric(
            "High/Critical Risk",
            high_risk
        )


    st.markdown("---")

    st.subheader(
        "🔬 Detection Pipeline"
    )

    st.markdown(
        """
        **NASA FIRMS**
        ↓

        **Thermal Anomaly Detection**
        ↓

        **Industrial Infrastructure Context**
        ↓

        **Land-Cover / Natural Context**
        ↓

        **Thermal Feature Engineering**
        ↓

        **Persistence Analysis**
        ↓

        **Random Forest Classification**
        ↓

        **Risk Scoring**
        ↓

        **GIS Visualization & Alerts**
        """
    )


# ============================================================
# GIS MAP
# ============================================================

elif page == "🗺️ Fire Risk Map":

    st.title(
        "🗺️ Industrial Fire & Thermal Risk Map"
    )

    if lat_col is None or lon_col is None:

        st.error(
            "Latitude/longitude data not found."
        )

        st.write(
            prediction_data.columns.tolist()
        )

        st.stop()


    data = prediction_data.copy()

    data[lat_col] = pd.to_numeric(
        data[lat_col],
        errors="coerce"
    )

    data[lon_col] = pd.to_numeric(
        data[lon_col],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            lat_col,
            lon_col
        ]
    )


    center_lat = data[
        lat_col
    ].mean()

    center_lon = data[
        lon_col
    ].mean()


    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    m = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=7,
        tiles=None
    )


    # --------------------------------------------------------
    # SATELLITE
    # --------------------------------------------------------

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="🛰️ Satellite Imagery"
    ).add_to(m)


    # --------------------------------------------------------
    # STREET
    # --------------------------------------------------------

    folium.TileLayer(
        "OpenStreetMap",
        name="🗺️ Street Map"
    ).add_to(m)


    # ========================================================
    # THERMAL HEATMAP
    # ========================================================

    heat_data = []


    if frp_col:

        data[frp_col] = pd.to_numeric(
            data[frp_col],
            errors="coerce"
        ).fillna(1)

        max_frp = data[
            frp_col
        ].max()

        if max_frp > 0:

            data["_intensity"] = (
                data[frp_col] /
                max_frp
            )

        else:

            data["_intensity"] = 1

    else:

        data["_intensity"] = 1


    for _, row in data.iterrows():

        heat_data.append(
            [
                float(row[lat_col]),
                float(row[lon_col]),
                float(row["_intensity"])
            ]
        )


    if heat_data:

        HeatMap(
            heat_data,
            name="🔥 Thermal Intensity",
            radius=28,
            blur=22,
            min_opacity=0.35,
            gradient={
                0.2: "blue",
                0.4: "cyan",
                0.6: "lime",
                0.75: "yellow",
                0.9: "orange",
                1.0: "red"
            }
        ).add_to(m)


    # ========================================================
    # CLASSIFICATION LAYERS
    # ========================================================

    categories = {

        "Industrial Fire": "red",

        "Persistent Industrial Thermal Source": "orange",

        "Natural / Forest Fire": "green",

        "Persistent Thermal Anomaly": "purple",

        "Other Thermal Anomaly": "blue"

    }


    for category, color in categories.items():

        group = folium.FeatureGroup(
            name=f"🔥 {category}",
            show=True
        )


        if classification_col:

            subset = data[
                data[
                    classification_col
                ]
                .astype(str)
                .str.lower()
                .str.contains(
                    category.lower(),
                    na=False
                )
            ]


            for _, row in subset.iterrows():

                popup = f"""
                <b>{category}</b><br><br>

                <b>FRP:</b>
                {row.get(frp_col, "N/A")}<br>

                <b>Risk:</b>
                {row.get(risk_col, "N/A")}<br>

                <b>Risk Score:</b>
                {row.get(risk_score_col, "N/A")}<br>

                <b>Land Cover:</b>
                {row.get("landcover", "Unknown")}<br>

                <b>Explanation:</b>
                {row.get("explanation", "N/A")}
                """


                folium.CircleMarker(

                    location=[
                        row[lat_col],
                        row[lon_col]
                    ],

                    radius=7,

                    color=color,

                    fill=True,

                    fill_color=color,

                    fill_opacity=0.9,

                    popup=folium.Popup(
                        popup,
                        max_width=350
                    )

                ).add_to(group)


        group.add_to(m)


    # ========================================================
    # INDUSTRIAL LOCATIONS
    # ========================================================

    industrial_group = folium.FeatureGroup(
        name="🏭 Industrial Infrastructure"
    )


    if not osm_data.empty:

        osm_lat = find_column(
            osm_data,
            [
                "latitude",
                "lat"
            ]
        )

        osm_lon = find_column(
            osm_data,
            [
                "longitude",
                "lon",
                "lng"
            ]
        )


        if osm_lat and osm_lon:

            for _, row in osm_data.iterrows():

                try:

                    lat = float(
                        row[osm_lat]
                    )

                    lon = float(
                        row[osm_lon]
                    )

                    folium.CircleMarker(

                        location=[
                            lat,
                            lon
                        ],

                        radius=5,

                        color="blue",

                        fill=True,

                        fill_color="blue",

                        fill_opacity=0.9,

                        popup="🏭 Industrial Infrastructure"

                    ).add_to(
                        industrial_group
                    )

                except:

                    pass


    industrial_group.add_to(m)


    folium.LayerControl(
        collapsed=False
    ).add_to(m)


    st_folium(
        m,
        height=700,
        width=None
    )


# ============================================================
# CLASSIFICATION
# ============================================================

elif page == "🔥 Fire Classification":

    st.title(
        "🔥 Thermal Source Classification"
    )

    if classification_col:

        counts = (
            prediction_data[
                classification_col
            ]
            .astype(str)
            .value_counts()
        )

        st.subheader(
            "Classification Summary"
        )

        st.bar_chart(
            counts
        )

        st.dataframe(
            prediction_data[
                [
                    classification_col,
                    risk_col,
                    risk_score_col,
                    "explanation"
                ]
            ],
            use_container_width=True
        )

    else:

        st.error(
            "Classification data not available."
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title(
        "📊 Thermal Analytics"
    )


    c1, c2 = st.columns(2)


    with c1:

        if classification_col:

            st.subheader(
                "Classification Distribution"
            )

            st.bar_chart(
                prediction_data[
                    classification_col
                ].value_counts()
            )


    with c2:

        if risk_col:

            st.subheader(
                "Risk Distribution"
            )

            st.bar_chart(
                prediction_data[
                    risk_col
                ].value_counts()
            )


    if frp_col:

        st.subheader(
            "🔥 FRP Distribution"
        )

        st.line_chart(
            pd.to_numeric(
                prediction_data[
                    frp_col
                ],
                errors="coerce"
            )
            .fillna(0)
            .reset_index(
                drop=True
            )
        )


# ============================================================
# ALERTS
# ============================================================

elif page == "🚨 Alerts":

    st.title(
        "🚨 Industrial Fire Alerts"
    )


    if risk_col:

        critical = prediction_data[
            prediction_data[
                risk_col
            ]
            .astype(str)
            .str.contains(
                "Critical",
                case=False,
                na=False
            )
        ]


        high = prediction_data[
            prediction_data[
                risk_col
            ]
            .astype(str)
            .str.contains(
                "High",
                case=False,
                na=False
            )
        ]


        st.error(
            f"🔴 Critical Alerts: {len(critical)}"
        )

        if len(critical):

            st.dataframe(
                critical,
                use_container_width=True
            )


        st.warning(
            f"🟠 High Risk Alerts: {len(high)}"
        )

        if len(high):

            st.dataframe(
                high,
                use_container_width=True
            )


# ============================================================
# AI MODEL
# ============================================================

elif page == "🤖 AI Model":

    st.title(
        "🤖 AI Classification Model"
    )


    if rf_model:

        st.success(
            "Random Forest model loaded."
        )


        st.write(
            "**Model:** Random Forest Classifier"
        )


        st.write(
            f"**Features:** {len(feature_columns)}"
        )


        if hasattr(
            rf_model,
            "feature_importances_"
        ):

            n = min(
                len(feature_columns),
                len(
                    rf_model.feature_importances_
                )
            )


            importance = pd.DataFrame({

                "Feature":
                    feature_columns[:n],

                "Importance":
                    rf_model
                    .feature_importances_[:n]

            })


            importance = (
                importance
                .sort_values(
                    "Importance",
                    ascending=False
                )
            )


            st.subheader(
                "Feature Importance"
            )


            st.bar_chart(
                importance.set_index(
                    "Feature"
                )
            )


            st.dataframe(
                importance,
                use_container_width=True
            )


    else:

        st.error(
            "Random Forest model unavailable."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SIH26162 | NASA FIRMS + OSM + GIS + "
    "Thermal Analysis + Machine Learning"
)
