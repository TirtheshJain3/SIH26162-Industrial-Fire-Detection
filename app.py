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

.metric-card {
    background: linear-gradient(135deg, #161b22, #21262d);
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
    font-size: 28px;
    font-weight: bold;
}

.alert-box {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_prediction_data():

    return pd.read_csv(
        "SIH26162_Prediction_Data.csv"
    )


@st.cache_data
def load_osm_data():

    try:
        return pd.read_csv(
            "SIH26162_OSM_Industrial_Data.csv"
        )

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
def load_features():

    try:
        return list(
            joblib.load(
                "SIH26162_Feature_List.pkl"
            )
        )

    except Exception:
        return []


# ============================================================
# LOAD
# ============================================================

try:

    prediction_data = load_prediction_data()
    osm_data = load_osm_data()
    rf_model = load_model()
    feature_columns = load_features()

except Exception as e:

    st.error("❌ Application data could not be loaded.")
    st.code(str(e))
    st.stop()


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(df, names):

    if df.empty:
        return None

    for name in names:

        for col in df.columns:

            if str(col).lower() == str(name).lower():
                return col

    return None


lat_col = find_column(
    prediction_data,
    ["latitude", "lat"]
)

lon_col = find_column(
    prediction_data,
    ["longitude", "lon", "lng"]
)

frp_col = find_column(
    prediction_data,
    ["frp", "frp_mw"]
)

classification_col = find_column(
    prediction_data,
    [
        "classification",
        "label",
        "predicted_class"
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

explanation_col = find_column(
    prediction_data,
    [
        "explanation",
        "reason",
        "classification_reason"
    ]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔥 SIH26162")

st.sidebar.caption(
    "Industrial Fire Intelligence System"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🗺️ Fire Risk Map",
        "🔥 Fire Classification",
        "📊 Analytics",
        "🚨 Alerts",
        "🤖 AI Model",
        "ℹ️ Data & Methodology"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "NASA FIRMS • OSM • GIS • Random Forest"
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
        ### AI-Based Detection, Classification and Monitoring
        of Industrial Fires and Persistent Thermal Sources
        """
    )

    st.markdown("---")

    total = len(prediction_data)

    if classification_col:

        classification_text = (
            prediction_data[
                classification_col
            ]
            .astype(str)
        )

        industrial_fire = (
            classification_text
            .str.contains(
                "Industrial Fire",
                case=False,
                na=False
            )
            .sum()
        )

        persistent_industrial = (
            classification_text
            .str.contains(
                "Persistent Industrial",
                case=False,
                na=False
            )
            .sum()
        )

        natural_fire = (
            classification_text
            .str.contains(
                "Natural",
                case=False,
                na=False
            )
            .sum()
        )

        other = (
            total
            - industrial_fire
            - persistent_industrial
            - natural_fire
        )

    else:

        industrial_fire = 0
        persistent_industrial = 0
        natural_fire = 0
        other = total


    if risk_col:

        risk_text = (
            prediction_data[
                risk_col
            ]
            .astype(str)
        )

        critical_high = (
            risk_text
            .str.contains(
                "Critical|High",
                case=False,
                na=False
            )
            .sum()
        )

    else:

        critical_high = 0


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
            persistent_industrial
        )

    with c4:
        st.metric(
            "Natural / Forest",
            natural_fire
        )

    with c5:
        st.metric(
            "High / Critical",
            critical_high
        )


    st.markdown("---")

    st.subheader(
        "🔬 Detection Pipeline"
    )

    st.markdown(
        """
        **NASA FIRMS**
        → Thermal Anomaly Detection
        → Industrial & Land Context
        → Feature Engineering
        → Persistence Analysis
        → Random Forest
        → Classification
        → Risk Scoring
        → GIS Visualization
        → Alerts
        """
    )

    st.markdown("---")

    st.subheader(
        "📊 Classification Distribution"
    )

    if classification_col:

        st.bar_chart(
            prediction_data[
                classification_col
            ].value_counts()
        )


# ============================================================
# FIRE RISK MAP
# ============================================================

elif page == "🗺️ Fire Risk Map":

    st.title(
        "🗺️ Industrial Fire & Thermal Risk Map"
    )

    st.caption(
        "Interactive GIS visualization of thermal intensity, "
        "industrial context, classification and risk."
    )


    # ========================================================
    # VALIDATE COORDINATES
    # ========================================================

    if lat_col is None or lon_col is None:

        st.error(
            "❌ Latitude/longitude columns were not found."
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


    # ========================================================
    # FILTERS
    # ========================================================

    st.subheader(
        "🎛️ Map Filters"
    )

    f1, f2, f3 = st.columns(3)


    # Classification filter
    with f1:

        if classification_col:

            available_classes = sorted(
                data[
                    classification_col
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_classes = st.multiselect(
                "Classification",
                available_classes,
                default=available_classes
            )

        else:

            selected_classes = []


    # Risk filter
    with f2:

        if risk_col:

            available_risks = [
                "Critical",
                "High",
                "Medium",
                "Low"
            ]

            existing_risks = [
                r for r in available_risks
                if r in
                data[
                    risk_col
                ]
                .astype(str)
                .unique()
            ]

            selected_risks = st.multiselect(
                "Risk Level",
                existing_risks,
                default=existing_risks
            )

        else:

            selected_risks = []


    # FRP filter
    with f3:

        if frp_col:

            data[frp_col] = pd.to_numeric(
                data[frp_col],
                errors="coerce"
            ).fillna(0)

            max_frp = float(
                data[frp_col].max()
            )

            min_frp = st.slider(
                "Minimum FRP",
                min_value=0.0,
                max_value=max(
                    max_frp,
                    1.0
                ),
                value=0.0
            )

        else:

            min_frp = 0


    # ========================================================
    # APPLY FILTERS
    # ========================================================

    filtered = data.copy()


    if classification_col and selected_classes:

        filtered = filtered[
            filtered[
                classification_col
            ]
            .astype(str)
            .isin(selected_classes)
        ]


    if risk_col and selected_risks:

        filtered = filtered[
            filtered[
                risk_col
            ]
            .astype(str)
            .isin(selected_risks)
        ]


    if frp_col:

        filtered = filtered[
            filtered[
                frp_col
            ] >= min_frp
        ]


    st.info(
        f"Showing **{len(filtered)}** "
        f"of **{len(data)}** detections."
    )


    # ========================================================
    # MAP CENTER
    # ========================================================

    if len(filtered) > 0:

        center_lat = filtered[
            lat_col
        ].mean()

        center_lon = filtered[
            lon_col
        ].mean()

    else:

        center_lat = data[
            lat_col
        ].mean()

        center_lon = data[
            lon_col
        ].mean()


    # ========================================================
    # CREATE MAP
    # ========================================================

    m = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=7,
        tiles=None,
        control_scale=True
    )


    # ========================================================
    # SATELLITE
    # ========================================================

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="🛰️ Satellite Imagery",
        overlay=False
    ).add_to(m)


    # ========================================================
    # STREET MAP
    # ========================================================

    folium.TileLayer(
        "OpenStreetMap",
        name="🗺️ Street Map",
        overlay=False
    ).add_to(m)


    # ========================================================
    # THERMAL HEATMAP
    # ========================================================

    heat_data = []


    if len(filtered) > 0:

        if frp_col:

            max_frp_filtered = max(
                float(
                    filtered[
                        frp_col
                    ].max()
                ),
                1
            )

            filtered[
                "_heat_intensity"
            ] = (
                filtered[
                    frp_col
                ] /
                max_frp_filtered
            )

        else:

            filtered[
                "_heat_intensity"
            ] = 1.0


        for _, row in filtered.iterrows():

            intensity = float(
                row[
                    "_heat_intensity"
                ]
            )

            intensity = max(
                0.1,
                min(
                    1.0,
                    intensity
                )
            )

            heat_data.append(
                [
                    float(
                        row[lat_col]
                    ),
                    float(
                        row[lon_col]
                    ),
                    intensity
                ]
            )


    if heat_data:

        HeatMap(
            heat_data,
            name="🔥 Thermal Intensity Heatmap",
            radius=30,
            blur=25,
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
    # CLASSIFICATION LAYERS
    # ========================================================

    layer_definitions = [

        (
            "Industrial Fire",
            "🔴 Industrial Fire",
            "red"
        ),

        (
            "Persistent Industrial Thermal Source",
            "🟠 Persistent Industrial Source",
            "orange"
        ),

        (
            "Natural / Forest Fire",
            "🟢 Natural / Forest Fire",
            "green"
        ),

        (
            "Persistent Thermal Anomaly",
            "🟣 Persistent Thermal Anomaly",
            "purple"
        ),

        (
            "Other Thermal Anomaly",
            "🔵 Other Thermal Anomaly",
            "blue"
        )
    ]


    for category, layer_name, color in layer_definitions:

        group = folium.FeatureGroup(
            name=layer_name,
            show=True
        )


        if classification_col:

            subset = filtered[
                filtered[
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

                classification = row.get(
                    classification_col,
                    "Unknown"
                )

                risk = (
                    row.get(
                        risk_col,
                        "Unknown"
                    )
                    if risk_col
                    else "Unknown"
                )

                score = (
                    row.get(
                        risk_score_col,
                        "N/A"
                    )
                    if risk_score_col
                    else "N/A"
                )

                frp_value = (
                    row.get(
                        frp_col,
                        "N/A"
                    )
                    if frp_col
                    else "N/A"
                )

                persistence = row.get(
                    "persistence_score",
                    "N/A"
                )

                detections = row.get(
                    "detections",
                    "N/A"
                )

                industrial_score = row.get(
                    "industrial_score",
                    "N/A"
                )

                explanation = (
                    row.get(
                        explanation_col,
                        "No explanation available."
                    )
                    if explanation_col
                    else "No explanation available."
                )


                popup_html = f"""
                <div style="
                    width:300px;
                    font-family:Arial;
                ">

                <h3>{layer_name}</h3>

                <hr>

                <b>Classification:</b>
                {classification}<br><br>

                <b>Risk Level:</b>
                {risk}<br><br>

                <b>Risk Score:</b>
                {score}<br><br>

                <b>FRP:</b>
                {frp_value}<br><br>

                <b>Persistence Score:</b>
                {persistence}<br><br>

                <b>Detections:</b>
                {detections}<br><br>

                <b>Industrial Score:</b>
                {industrial_score}<br><br>

                <b>AI Evidence:</b><br>
                {explanation}<br><br>

                <b>Latitude:</b>
                {float(row[lat_col]):.5f}<br>

                <b>Longitude:</b>
                {float(row[lon_col]):.5f}

                </div>
                """


                folium.CircleMarker(

                    location=[
                        float(
                            row[lat_col]
                        ),
                        float(
                            row[lon_col]
                        )
                    ],

                    radius=7,

                    color=color,

                    fill=True,

                    fill_color=color,

                    fill_opacity=0.9,

                    weight=2,

                    popup=folium.Popup(
                        popup_html,
                        max_width=350
                    )

                ).add_to(group)


        group.add_to(m)


    # ========================================================
    # INDUSTRIAL INFRASTRUCTURE
    # ========================================================

    industrial_group = folium.FeatureGroup(
        name="🏭 Industrial Infrastructure",
        show=True
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

            osm_copy = osm_data.copy()

            osm_copy[
                osm_lat
            ] = pd.to_numeric(
                osm_copy[
                    osm_lat
                ],
                errors="coerce"
            )

            osm_copy[
                osm_lon
            ] = pd.to_numeric(
                osm_copy[
                    osm_lon
                ],
                errors="coerce"
            )

            osm_copy = osm_copy.dropna(
                subset=[
                    osm_lat,
                    osm_lon
                ]
            )


            for _, row in osm_copy.iterrows():

                folium.CircleMarker(

                    location=[
                        float(
                            row[osm_lat]
                        ),
                        float(
                            row[osm_lon]
                        )
                    ],

                    radius=5,

                    color="blue",

                    fill=True,

                    fill_color="blue",

                    fill_opacity=0.9,

                    popup=folium.Popup(
                        "<b>🏭 Industrial Infrastructure</b>",
                        max_width=250
                    )

                ).add_to(
                    industrial_group
                )


    industrial_group.add_to(m)


    # ========================================================
    # MAP LEGEND
    # ========================================================

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: white;
        padding: 12px;
        border-radius: 8px;
        border: 2px solid grey;
        font-size: 13px;
    ">

    <b>🔥 Classification</b><br>

    <span style="color:red;">●</span>
    Industrial Fire<br>

    <span style="color:orange;">●</span>
    Persistent Industrial<br>

    <span style="color:green;">●</span>
    Natural / Forest<br>

    <span style="color:purple;">●</span>
    Persistent Thermal<br>

    <span style="color:blue;">●</span>
    Other Thermal<br>

    <span style="color:#0000ff;">●</span>
    Industrial Location

    </div>
    """

    m.get_root().html.add_child(
        folium.Element(
            legend_html
        )
    )


    # ========================================================
    # LAYER CONTROL
    # ========================================================

    folium.LayerControl(
        collapsed=False
    ).add_to(m)


    # ========================================================
    # DISPLAY
    # ========================================================

    st_folium(
        m,
        width=None,
        height=700,
        returned_objects=[]
    )


# ============================================================
# FIRE CLASSIFICATION
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

        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                "Classification Distribution"
            )

            st.bar_chart(
                counts
            )

        with c2:

            st.subheader(
                "Classification Counts"
            )

            st.dataframe(
                counts.rename(
                    "Detections"
                ),
                use_container_width=True
            )


        st.markdown("---")

        st.subheader(
            "🔎 Detection Details"
        )

        display_columns = []

        for col in [
            lat_col,
            lon_col,
            classification_col,
            risk_col,
            risk_score_col,
            frp_col,
            "persistence_score",
            "detections",
            "industrial_score",
            explanation_col
        ]:

            if col and col in prediction_data.columns:
                if col not in display_columns:
                    display_columns.append(col)


        st.dataframe(
            prediction_data[
                display_columns
            ],
            use_container_width=True,
            height=550
        )

    else:

        st.error(
            "Classification column not available."
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
                "Classification"
            )

            st.bar_chart(
                prediction_data[
                    classification_col
                ].value_counts()
            )


    with c2:

        if risk_col:

            st.subheader(
                "Risk Levels"
            )

            st.bar_chart(
                prediction_data[
                    risk_col
                ].value_counts()
            )


    if frp_col:

        st.subheader(
            "🔥 Fire Radiative Power"
        )

        frp_series = pd.to_numeric(
            prediction_data[
                frp_col
            ],
            errors="coerce"
        ).fillna(0)

        st.line_chart(
            frp_series.reset_index(
                drop=True
            )
        )


    if risk_score_col:

        st.subheader(
            "📈 Risk Score Distribution"
        )

        score_series = pd.to_numeric(
            prediction_data[
                risk_score_col
            ],
            errors="coerce"
        ).fillna(0)

        st.bar_chart(
            score_series
            .value_counts()
            .sort_index()
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


        medium = prediction_data[
            prediction_data[
                risk_col
            ]
            .astype(str)
            .str.contains(
                "Medium",
                case=False,
                na=False
            )
        ]


        c1, c2, c3 = st.columns(3)

        with c1:
            st.error(
                f"🔴 Critical\n\n{len(critical)}"
            )

        with c2:
            st.warning(
                f"🟠 High\n\n{len(high)}"
            )

        with c3:
            st.info(
                f"🟡 Medium\n\n{len(medium)}"
            )


        st.markdown("---")


        if len(critical) > 0:

            st.subheader(
                "🔴 Critical Alerts"
            )

            st.dataframe(
                critical,
                use_container_width=True
            )


        if len(high) > 0:

            st.subheader(
                "🟠 High Risk Alerts"
            )

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


    if rf_model is not None:

        st.success(
            "✅ Random Forest model loaded successfully."
        )


        c1, c2 = st.columns(2)


        with c1:

            st.metric(
                "Model",
                "Random Forest"
            )


        with c2:

            st.metric(
                "Features",
                len(feature_columns)
            )


        st.subheader(
            "Input Features"
        )

        for feature in feature_columns:

            st.write(
                f"• {feature}"
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
                "📈 Feature Importance"
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
            "Random Forest model could not be loaded."
        )


# ============================================================
# DATA & METHODOLOGY
# ============================================================

elif page == "ℹ️ Data & Methodology":

    st.title(
        "ℹ️ Data Sources & Methodology"
    )


    st.subheader(
        "🛰️ Data Sources"
    )

    st.markdown(
        """
        **NASA FIRMS**
        - Satellite-based thermal anomaly/fire detections
        - FRP and brightness information

        **OpenStreetMap**
        - Industrial infrastructure
        - Industrial areas and facilities

        **Satellite Basemap**
        - GIS visualization and spatial interpretation

        **Machine Learning**
        - Random Forest classification

        **GIS**
        - Spatial relationship between thermal anomalies
          and industrial infrastructure
        """
    )


    st.markdown("---")


    st.subheader(
        "🧠 Processing Pipeline"
    )

    st.code(
        """
NASA FIRMS
     ↓
Thermal Anomaly Detection
     ↓
Industrial Infrastructure Context
     ↓
Land / Natural Context
     ↓
Thermal Feature Engineering
     ↓
Persistence Analysis
     ↓
Random Forest Classification
     ↓
4-Class Classification
     ↓
Risk Score
     ↓
GIS Visualization
     ↓
Alerts
        """
    )


    st.subheader(
        "🎯 Classification Categories"
    )

    st.markdown(
        """
        🔴 **Industrial Fire**

        Thermal event associated with industrial infrastructure.

        🟠 **Persistent Industrial Thermal Source**

        Repeated or persistent thermal activity associated with
        industrial infrastructure.

        🟢 **Natural / Forest Fire**

        Thermal activity associated with natural or vegetated
        areas.

        🔵 **Other Thermal Anomaly**

        Thermal detection without sufficient evidence for the
        other categories.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SIH26162 | AI-Based Detection and Classification of "
    "Industrial Fires and Persistent Thermal Sources"
)

st.caption(
    "NASA FIRMS • OpenStreetMap • GIS • Random Forest • Streamlit"
)
