
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium

from streamlit_folium import st_folium

from folium.plugins import HeatMap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SIH26162 Fire Intelligence",
    page_icon="🔥",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
    text-align: center;
}

.alert-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #ffe5e5;
    border-left: 5px solid #ff0000;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    prediction_data = pd.read_csv(
        "SIH26162_Prediction_Data.csv"
    )

    osm_data = pd.read_csv(
        "SIH26162_OSM_Industrial_Data.csv"
    )

    return prediction_data, osm_data


@st.cache_resource
def load_model():

    model = joblib.load(
        "SIH26162_RandomForest_Model.pkl"
    )

    features = joblib.load(
        "SIH26162_Feature_List.pkl"
    )

    return model, features


prediction_data, osm_data = load_data()

rf_model, feature_columns = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔥 SIH26162")

st.sidebar.markdown(
    """
### AI Industrial Fire Monitoring

NASA FIRMS + OSM + Machine Learning
"""
)

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


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "🔥 AI-Based Industrial Fire Intelligence"
    )

    st.markdown(
        "### SIH26162"
    )

    st.write(
        "Detection and classification of industrial fires "
        "and persistent thermal sources using NASA FIRMS, "
        "OpenStreetMap and Machine Learning."
    )

    st.divider()

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total = len(prediction_data)

    industrial = (
        prediction_data["classification"]
        .eq("Industrial Fire")
        .sum()
    )

    persistent = (
        prediction_data["classification"]
        .eq("Persistent Thermal Source")
        .sum()
    )

    high_risk = (
        prediction_data["risk_category"]
        .isin(["HIGH", "CRITICAL"])
        .sum()
    )

    avg_confidence = (
        prediction_data[
            "prediction_confidence"
        ].mean() * 100
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Thermal Detections",
        total
    )

    col2.metric(
        "Industrial Fires",
        industrial
    )

    col3.metric(
        "Persistent Sources",
        persistent
    )

    col4.metric(
        "High/Critical Risk",
        high_risk
    )

    col5.metric(
        "AI Confidence",
        f"{avg_confidence:.1f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # RECENT/HIGHEST RISK DETECTIONS
    # --------------------------------------------------------

    st.subheader(
        "Highest Risk Thermal Detections"
    )

    display_columns = [
        "latitude",
        "longitude",
        "classification",
        "risk_score",
        "risk_category",
        "frp",
        "distance_km",
        "persistence_score",
        "prediction_confidence"
    ]

    st.dataframe(
        prediction_data[
            display_columns
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10),
        use_container_width=True
    )


# ============================================================
# MAP
# ============================================================

elif page == "🗺️ Fire Risk Map":

    st.title(
        "🗺️ Interactive Fire Risk Map"
    )

    st.write(
        "NASA FIRMS thermal detections with "
        "industrial proximity, AI classification and risk."
    )

    map_center = [
        prediction_data["latitude"].mean(),
        prediction_data["longitude"].mean()
    ]

    m = folium.Map(
        location=map_center,
        zoom_start=9,
        tiles=None
    )

    # Satellite
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Satellite",
        overlay=False
    ).add_to(m)

    # Street
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Street_Map/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Street",
        overlay=False
    ).add_to(m)

    # --------------------------------------------------------
    # INDUSTRIAL LOCATIONS
    # --------------------------------------------------------

    industrial_layer = folium.FeatureGroup(
        name="🏭 Industrial Locations"
    )

    for _, row in osm_data.iterrows():

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=4,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.7,
            popup="Industrial Location"
        ).add_to(
            industrial_layer
        )

    industrial_layer.add_to(m)

    # --------------------------------------------------------
    # FIRE DETECTIONS
    # --------------------------------------------------------

    for _, row in prediction_data.iterrows():

        if row["classification"] == "Industrial Fire":

            color = "red"

        elif row["classification"] == "Persistent Thermal Source":

            color = "orange"

        else:

            color = "green"

        popup = f"""
        <b>🔥 AI Fire Detection</b><br><br>

        Classification:
        {row["classification"]}<br>

        Risk:
        {row["risk_category"]}<br>

        Risk Score:
        {row["risk_score"]:.1f}/100<br>

        AI Confidence:
        {row["prediction_confidence"]*100:.1f}%<br>

        FRP:
        {row["frp"]:.2f} MW<br>

        Industrial Distance:
        {row["distance_km"]:.2f} km<br>

        Persistence:
        {row["persistence_score"]:.1f}/100
        """

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(
                popup,
                max_width=300
            )
        ).add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(
        m,
        width=None,
        height=650
    )


# ============================================================
# FIRE DETECTIONS
# ============================================================

elif page == "🔥 Fire Detections":

    st.title(
        "🔥 Fire Detection Database"
    )

    # Filters

    classification_filter = st.multiselect(
        "Classification",
        prediction_data[
            "classification"
        ].unique(),
        default=list(
            prediction_data[
                "classification"
            ].unique()
        )
    )

    risk_filter = st.multiselect(
        "Risk Category",
        prediction_data[
            "risk_category"
        ].unique(),
        default=list(
            prediction_data[
                "risk_category"
            ].unique()
        )
    )

    filtered = prediction_data[
        prediction_data[
            "classification"
        ].isin(classification_filter)
        &
        prediction_data[
            "risk_category"
        ].isin(risk_filter)
    ]

    st.write(
        f"Showing {len(filtered)} detections"
    )

    st.dataframe(
        filtered,
        use_container_width=True
    )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title(
        "📊 Fire Analytics"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Classification Distribution"
        )

        st.bar_chart(
            prediction_data[
                "classification"
            ].value_counts()
        )

    with col2:

        st.subheader(
            "Risk Distribution"
        )

        st.bar_chart(
            prediction_data[
                "risk_category"
            ].value_counts()
        )

    st.subheader(
        "FRP Distribution"
    )

    st.line_chart(
        prediction_data[
            "frp"
        ].sort_values()
        .reset_index(drop=True)
    )

    st.subheader(
        "Risk vs Industrial Distance"
    )

    st.scatter_chart(
        prediction_data,
        x="distance_km",
        y="risk_score"
    )


# ============================================================
# ALERTS
# ============================================================

elif page == "🚨 Alerts":

    st.title(
        "🚨 Fire Risk Alerts"
    )

    alerts = prediction_data[
        prediction_data[
            "risk_category"
        ].isin(
            ["HIGH", "CRITICAL"]
        )
    ].sort_values(
        "risk_score",
        ascending=False
    )

    if len(alerts) == 0:

        st.success(
            "No HIGH or CRITICAL risk detections."
        )

    else:

        st.error(
            f"{len(alerts)} HIGH/CRITICAL "
            "risk detections identified."
        )

        for _, row in alerts.head(10).iterrows():

            st.markdown(
                f"""
                ### 🔥 {row["risk_category"]}

                **Classification:** {row["classification"]}

                **Risk Score:** {row["risk_score"]:.1f}/100

                **AI Confidence:** {row["prediction_confidence"]*100:.1f}%

                **FRP:** {row["frp"]:.2f} MW

                **Industrial Distance:** {row["distance_km"]:.2f} km

                **Coordinates:**
                {row["latitude"]:.5f},
                {row["longitude"]:.5f}
                """
            )

            st.divider()


# ============================================================
# AI MODEL
# ============================================================

elif page == "🤖 AI Model":

    st.title(
        "🤖 AI Model Information"
    )

    st.subheader(
        "Random Forest"
    )

    st.write(
        "The model uses thermal, spatial and persistence "
        "features to classify thermal detections."
    )

    st.subheader(
        "Features"
    )

    for feature in feature_columns:

        st.write(
            f"• {feature}"
        )

    st.subheader(
        "Model Parameters"
    )

    st.write(
        f"Number of Trees: "
        f"{rf_model.n_estimators}"
    )

    st.write(
        f"Maximum Depth: "
        f"{rf_model.max_depth}"
    )

    st.subheader(
        "Feature Importance"
    )

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance":
            rf_model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
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
