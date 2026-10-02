import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

FEATURE_LABELS = {
    "rolling_3_month_delay_rate": "Average delay rate (last 3 months)",
    "rolling_6_month_delay_rate": "Average delay rate (last 6 months)",
    "rolling_12_month_delay_rate": "Average delay rate (last 12 months)",
    "lag_1_delay_rate": "Delay rate in previous month",
    "lag_2_delay_rate": "Delay rate 2 months earlier",
    "lag_3_delay_rate": "Delay rate 3 months earlier",
    "lag_12_delay_rate": "Delay rate 12 months earlier",
    "previous_month_avg_delay": "Average delay minutes in previous month",
    "previous_month_cancel_rate": "Cancellation rate in previous month",
    "previous_month_arr_flights": "Flights arriving in previous month",
    "previous_month_diversion_rate": "Diversion rate in previous month",
    "month": "Calendar month",
    "month_sin": "Seasonality (sine)",
    "month_cos": "Seasonality (cosine)",
}


def readable_feature_name(feature):
    """Convert model feature names into labels users can understand."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if str(feature).startswith("cat_"):
        return "Encoded airline/airport category"
    return str(feature).replace("_", " ").capitalize()


# --- Page Configuration ---
st.set_page_config(
    page_title="Airline Flight Delay Prediction Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Glassmorphism & Sleek Dark Mode UI) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.3);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 8px;
    }
    
    .metric-subtitle {
        color: #38bdf8;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    .badge-high {
        background-color: #ef4444;
        color: white;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .badge-low {
        background-color: #10b981;
        color: white;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading Helpers ---
@st.cache_data
def load_data():
    analytics_dir = "output/analytics"
    preds_dir = "output/predictions"
    reports_dir = "output/reports"

    airline_rankings = pd.read_csv(f"{analytics_dir}/airline_rankings.csv") if os.path.exists(f"{analytics_dir}/airline_rankings.csv") else pd.DataFrame()
    airport_rankings = pd.read_csv(f"{analytics_dir}/airport_rankings.csv") if os.path.exists(f"{analytics_dir}/airport_rankings.csv") else pd.DataFrame()
    delay_causes = pd.read_csv(f"{analytics_dir}/delay_causes.csv") if os.path.exists(f"{analytics_dir}/delay_causes.csv") else pd.DataFrame()
    monthly_trends = pd.read_csv(f"{analytics_dir}/monthly_trends.csv") if os.path.exists(f"{analytics_dir}/monthly_trends.csv") else pd.DataFrame()
    annual_trends = pd.read_csv(f"{analytics_dir}/annual_trends.csv") if os.path.exists(f"{analytics_dir}/annual_trends.csv") else pd.DataFrame()
    poor_combos = pd.read_csv(f"{analytics_dir}/poor_performing_combinations.csv") if os.path.exists(f"{analytics_dir}/poor_performing_combinations.csv") else pd.DataFrame()
    predictions = pd.read_csv(f"{preds_dir}/predictions.csv") if os.path.exists(f"{preds_dir}/predictions.csv") else pd.DataFrame()
    
    reg_results = pd.read_csv(f"{reports_dir}/regression_results.csv") if os.path.exists(f"{reports_dir}/regression_results.csv") else pd.DataFrame()
    clf_results = pd.read_csv(f"{reports_dir}/classification_results.csv") if os.path.exists(f"{reports_dir}/classification_results.csv") else pd.DataFrame()
    feat_imp = pd.read_csv(f"{reports_dir}/feature_importance.csv") if os.path.exists(f"{reports_dir}/feature_importance.csv") else pd.DataFrame()

    if not delay_causes.empty:
        if "total_delay_hours" not in delay_causes.columns and "total_delay_minutes" in delay_causes.columns:
            delay_causes["total_delay_hours"] = delay_causes["total_delay_minutes"] / 60
        delay_causes["delay_cause"] = delay_causes["delay_cause"].replace({
            "Late Aircraft Delay": "Late Arrival Delay"
        })

    if not monthly_trends.empty and "month" in monthly_trends.columns:
        monthly_trends["month_name"] = monthly_trends["month"].map(MONTH_NAMES)
        monthly_trends["month_name"] = pd.Categorical(
            monthly_trends["month_name"],
            categories=list(MONTH_NAMES.values()),
            ordered=True
        )

    return {
        "airline_rankings": airline_rankings,
        "airport_rankings": airport_rankings,
        "delay_causes": delay_causes,
        "monthly_trends": monthly_trends,
        "annual_trends": annual_trends,
        "poor_combos": poor_combos,
        "predictions": predictions,
        "reg_results": reg_results,
        "clf_results": clf_results,
        "feat_imp": feat_imp
    }

data = load_data()

# --- Sidebar Navigation ---
st.sidebar.image("https://img.icons8.com/isometric-line/100/4f46e5/airplane-take-off.png", width=70)
st.sidebar.title("Flight Delay Portal")
st.sidebar.caption("Hadoop HDFS + Apache PySpark ML Engine")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Airline Analysis", "Airport Analysis", "Delay Predictions", "Model Evaluation & Metrics"]
)

st.sidebar.markdown("---")
st.sidebar.info("System Status: **Connected to Storage**\nArchitecture: **Distributed PySpark ML**")

# ==========================================
# PAGE 1: OVERVIEW METRICS
# ==========================================
if page == "Overview":
    st.title("Executive Flight Delay Overview")
    st.caption("Real-time distributed analytics on airline performance metrics")

    # High-level KPIs
    tot_flights = data["annual_trends"]["total_flights"].sum() if not data["annual_trends"].empty else 0
    avg_delay_rate = data["airline_rankings"]["delay_rate"].mean() if not data["airline_rankings"].empty else 0.0
    avg_cancel_rate = data["airline_rankings"]["cancellation_rate"].mean() if not data["airline_rankings"].empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Arriving Flights</div>
            <div class="metric-value">{tot_flights:,.0f}</div>
            <div class="metric-subtitle">Historical Dataset Scope</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Delay Rate</div>
            <div class="metric-value">{avg_delay_rate*100:.2f}%</div>
            <div class="metric-subtitle">Arrivals Delayed > 15m</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Cancellation Rate</div>
            <div class="metric-value">{avg_cancel_rate*100:.2f}%</div>
            <div class="metric-subtitle">Flight Cancellations</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tracked Airlines</div>
            <div class="metric-value">{len(data['airline_rankings'])}</div>
            <div class="metric-subtitle">Active Commercial Carriers</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Annual Delay & Cancellation Trends")
        if not data["annual_trends"].empty:
            fig = px.line(
                data["annual_trends"],
                x="year",
                y=["annual_delay_rate", "annual_cancellation_rate"],
                labels={"value": "Rate", "year": "Year", "variable": "Metric"},
                markers=True,
                color_discrete_sequence=["#6366f1", "#ec4899"]
            )
            fig.update_layout(template="plotly_dark", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Dominant Delay Causes (Total Hours)")
        if not data["delay_causes"].empty:
            fig = px.pie(
                data["delay_causes"],
                names="delay_cause",
                values="total_delay_hours",
                labels={"total_delay_hours": "Total Hours"},
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(template="plotly_dark", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

# ==========================================
# PAGE 2: AIRLINE ANALYSIS
# ==========================================
elif page == "Airline Analysis":
    st.title("Airline Deep-Dive Analysis")
    
    if not data["airline_rankings"].empty:
        carriers = sorted(data["airline_rankings"]["carrier_name"].unique())
        selected_carrier_name = st.selectbox("Select Airline Carrier", carriers)

        carrier_info = data["airline_rankings"][data["airline_rankings"]["carrier_name"] == selected_carrier_name].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Flights", f"{carrier_info['total_flights']:,.0f}")
        with col2:
            st.metric("Overall Delay Rate", f"{carrier_info['delay_rate']*100:.2f}%")
        with col3:
            st.metric("Cancellation Rate", f"{carrier_info['cancellation_rate']*100:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top Airports Served by Carrier (Worst Delay Rates)")
            if not data["poor_combos"].empty:
                carrier_combos = data["poor_combos"][data["poor_combos"]["carrier_name"] == selected_carrier_name].head(10)
                if not carrier_combos.empty:
                    fig = px.bar(carrier_combos, x="delay_rate", y="airport_name", orientation="h", color="delay_rate", color_continuous_scale="Reds")
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No route-specific breakdown available for selected carrier.")
        with c2:
            st.subheader("Monthly Seasonality Pattern")
            if not data["monthly_trends"].empty:
                fig = px.bar(data["monthly_trends"], x="month_name", y="delay_rate", color="delay_rate", color_continuous_scale="Purples")
                fig.update_xaxes(categoryorder="array", categoryarray=list(MONTH_NAMES.values()))
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")

# ==========================================
# PAGE 3: AIRPORT ANALYSIS
# ==========================================
elif page == "Airport Analysis":
    st.title("Airport Operations Analysis")

    if not data["airport_rankings"].empty:
        airports = sorted(data["airport_rankings"]["airport_name"].unique())
        selected_airport_name = st.selectbox("Select Airport", airports)

        ap_info = data["airport_rankings"][data["airport_rankings"]["airport_name"] == selected_airport_name].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Arrival Flights", f"{ap_info['total_flights']:,.0f}")
        with col2:
            st.metric("Airport Delay Rate", f"{ap_info['delay_rate']*100:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Carriers Operating at Airport (Delay Rate Breakdown)")
        if not data["poor_combos"].empty:
            ap_combos = data["poor_combos"][data["poor_combos"]["airport_name"] == selected_airport_name]
            if not ap_combos.empty:
                fig = px.bar(ap_combos, x="carrier_name", y="delay_rate", color="delay_rate", color_continuous_scale="Viridis")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Operating carrier breakdown details unavailable for this airport.")

# ==========================================
# PAGE 4: DELAY PREDICTIONS
# ==========================================
elif page == "Delay Predictions":
    st.title("Predictive ML Delay Rate Engine")
    st.caption("Forecast future airline-airport delay rate and high-delay probability")

    if not data["predictions"].empty:
        st.subheader("Generate Future Route Prediction")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            predictions = data["predictions"]
            carrier_names = (
                predictions.drop_duplicates("carrier")
                .set_index("carrier")["carrier_name"]
                .to_dict()
                if "carrier_name" in predictions.columns
                else {}
            )
            carriers = sorted(
                predictions["carrier"].dropna().unique(),
                key=lambda code: str(carrier_names.get(code, code)).casefold()
            ) if "carrier" in predictions.columns else ["AA", "DL", "UA"]
            sel_carrier = st.selectbox(
                "Airline",
                carriers,
                format_func=lambda code: f"{carrier_names.get(code, 'Unknown airline')} [{code}]"
            )
        with c2:
            carrier_predictions = predictions[predictions["carrier"] == sel_carrier]
            airport_names = (
                carrier_predictions.drop_duplicates("airport")
                .set_index("airport")["airport_name"]
                .to_dict()
                if "airport_name" in carrier_predictions.columns
                else {}
            )
            airports = sorted(
                carrier_predictions["airport"].dropna().unique(),
                key=lambda code: str(airport_names.get(code, code)).casefold()
            ) if "airport" in carrier_predictions.columns else []

            if airports:
                sel_airport = st.selectbox(
                    "Airport",
                    airports,
                    format_func=lambda code: f"{airport_names.get(code, 'Unknown airport')} [{code}]"
                )
            else:
                sel_airport = None
                st.warning("No airport predictions are available for the selected airline.")
        with c3:
            target_month = st.selectbox(
                "Prediction Target Month",
                list(range(1, 13)),
                format_func=lambda month: MONTH_NAMES[month]
            )

        # Filter or compute prediction output
        match_pred = data["predictions"][
            (data["predictions"]["carrier"] == sel_carrier) &
            (data["predictions"]["airport"] == sel_airport) &
            (data["predictions"]["prediction_month"] == target_month)
        ]

        if not match_pred.empty:
            pred_row = match_pred.iloc[0]
            pred_val = float(pred_row["predicted_delay_rate"])
            high_delay_flag = int(pred_row["predicted_high_delay"])
            prediction_available = True
        else:
            pred_val = None
            high_delay_flag = None
            prediction_available = False

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if not prediction_available:
                if sel_airport is None:
                    st.warning(
                        f"No airport predictions are available for "
                        f"{carrier_names.get(sel_carrier, 'the selected airline')}."
                    )
                else:
                    st.warning(
                        f"No prediction is available for {sel_carrier} at "
                        f"{sel_airport} for {MONTH_NAMES[target_month]}."
                    )
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Predicted Delay Rate</div>
                    <div class="metric-value">{pred_val*100:.2f}%</div>
                    <div class="metric-subtitle">Forecasted for {carrier_names.get(sel_carrier, 'Unknown airline')} [{sel_carrier}] at {airport_names.get(sel_airport, 'Unknown airport')} [{sel_airport}]</div>
                </div>
                """, unsafe_allow_html=True)
        with col2:
            if prediction_available:
                status_badge = '<span class="badge-high">HIGH DELAY RISK (≥ 20%)</span>' if high_delay_flag == 1 else '<span class="badge-low">NORMAL DELAY RISK (< 20%)</span>'
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">High Delay Classification</div>
                    <div style="margin-top: 14px;">{status_badge}</div>
                    <div class="metric-subtitle">ML Binary Classification Model Output</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Upload Prediction CSV")
        st.info(
            "No predictions appear in this box until you upload a CSV. "
            "The dashboard displays precomputed Spark model outputs; it does not retrain or create new predictions from an empty upload. "
            "Use the example CSV below to preview the expected format."
        )
        uploaded_predictions = st.file_uploader(
            "Drag and drop a prediction CSV here",
            type=["csv"],
            help="Upload a CSV containing carrier, airport, prediction_month, and predicted_delay_rate columns."
        )

        if uploaded_predictions is None:
            st.caption("Prediction preview is empty because no CSV has been uploaded yet.")
        else:
            uploaded_data = pd.read_csv(uploaded_predictions)
            uploaded_data = uploaded_data.drop(columns=["prediction_year", "prediction_month"], errors="ignore")
            st.success(f"Loaded {len(uploaded_data):,} prediction rows from {uploaded_predictions.name}.")

            def delay_row_style(row):
                rate = pd.to_numeric(row.get("predicted_delay_rate"), errors="coerce")
                if pd.isna(rate):
                    return [""] * len(row)
                if rate >= 0.20:
                    color = "#7f1d1d"
                elif rate >= 0.15:
                    color = "#9a3412"
                else:
                    color = "#166534"
                return [f"background-color: {color}; color: white"] * len(row)

            st.dataframe(uploaded_data.style.apply(delay_row_style, axis=1), width="stretch")

# ==========================================
# PAGE 5: MODEL EVALUATION & METRICS
# ==========================================
elif page == "Model Evaluation & Metrics":
    st.title("Machine Learning Model Evaluation")
    st.caption("These results show how accurately the models estimate delay rates and identify high-delay routes.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Regression Models Comparison")
        st.caption("Regression predicts the delay rate as a percentage. Lower RMSE and MAE are better; higher R² is better.")
        if not data["reg_results"].empty:
            regression_display = data["reg_results"].rename(columns={
                "Model": "Model",
                "Validation_RMSE": "Validation error (RMSE)",
                "Validation_MAE": "Validation average error (MAE)",
                "Validation_R2": "Validation explained variation (R²)",
                "Test_RMSE": "Test error (RMSE)",
                "Test_MAE": "Test average error (MAE)",
                "Test_R2": "Test explained variation (R²)",
            })
            st.dataframe(regression_display, width="stretch", hide_index=True)
    with col2:
        st.subheader("Classification Models Comparison")
        st.caption("Classification identifies whether a route has high delay risk. Accuracy, F1, and ROC-AUC are higher-is-better scores.")
        if not data["clf_results"].empty:
            classification_display = data["clf_results"].rename(columns={
                "Validation_Accuracy": "Validation accuracy",
                "Validation_F1": "Validation F1 score",
                "Validation_ROC_AUC": "Validation ROC-AUC",
                "Test_Accuracy": "Test accuracy",
                "Test_Precision": "Test precision",
                "Test_Recall": "Test recall",
                "Test_F1": "Test F1 score",
                "Test_ROC_AUC": "Test ROC-AUC",
            })
            st.dataframe(classification_display, width="stretch", hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Evaluation Artifact Charts")

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("output/charts/model_comparison.png"):
            st.image("output/charts/model_comparison.png", caption="Model Performance Comparison", width="stretch")
        if os.path.exists("output/charts/confusion_matrix.png"):
            st.image("output/charts/confusion_matrix.png", caption="Classification Confusion Matrix", width="stretch")

    with c2:
        if os.path.exists("output/charts/actual_vs_predicted.png"):
            st.image("output/charts/actual_vs_predicted.png", caption="Actual vs Predicted Delay Rates", width="stretch")
        if not data["feat_imp"].empty:
            feature_chart_data = data["feat_imp"].head(10).copy()
            feature_chart_data["Feature"] = feature_chart_data["Feature"].map(readable_feature_name)
            feature_chart = px.bar(
                feature_chart_data.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                labels={
                    "Importance": "Contribution to model decision",
                    "Feature": "Prediction input",
                },
                color="Importance",
                color_continuous_scale="Magma",
                title="Top 10 Factors Influencing Delay-Rate Predictions",
            )
            feature_chart.update_layout(
                template="plotly_dark",
                height=430,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(feature_chart, width="stretch")
            st.caption("Importance shows how much each input contributes to the tree model's decisions. It does not indicate whether the factor increases or decreases delays.")
