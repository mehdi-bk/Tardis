import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TARDIS — SNCF Delay Dashboard",
    page_icon="🚄",
    layout="wide",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem;}
    .metric-card {
        background: #1e2535;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 3px solid #3b82f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")

    # Station name standardization (same as model notebook)
    reps = {
        "ANGERS ST LAUD": "ANGERS SAINT LAUD",
        "ST ETIENNE CHATEAUCREUX": "SAINT ETIENNE CHATEAUCREUX",
        "BORDEAUX ST JEAN": "BORDEAUX SAINT JEAN",
        "MARSEILLE ST CHARLES": "MARSEILLE SAINT CHARLES",
        "ST MALO": "SAINT MALO",
        "ST PIERRE DES CORPS": "SAINT PIERRE DES CORPS",
    }
    df["Gare de départ"] = df["Gare de départ"].str.strip().str.upper().replace(reps)
    df["Gare d'arrivée"] = df["Gare d'arrivée"].str.strip().str.upper().replace(reps)
    df["Service"] = df["Service"].str.strip().str.upper()

    # Outlier removal on target (same as model notebook)
    target = "Retard moyen de tous les trains à l'arrivée"
    df[target] = pd.to_numeric(df[target], errors="coerce")
    q_low = df[target].quantile(0.01)
    q_high = df[target].quantile(0.99)
    df = df[(df[target] >= q_low) & (df[target] <= q_high)]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    df = df.dropna(subset=["Gare de départ", "Gare d'arrivée"])

    return df


# ── LOAD MODEL & FEATURES ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    feature_cols = joblib.load("model_features.joblib")
    return model, feature_cols


# ── ROUTE AVERAGES (for smart prediction filling) ─────────────────────────────
@st.cache_data
def compute_route_averages(df):
    """Compute historical averages per route for all numeric features."""
    numeric_cols = [
        "Durée moyenne du trajet",
        "Nombre de circulations prévues",
        "Nombre de trains annulés",
        "Nombre de trains en retard au départ",
        "Retard moyen des trains en retard au départ",
        "Retard moyen de tous les trains au départ",
        "Nombre trains en retard > 15min",
        "Nombre trains en retard > 30min",
        "Nombre trains en retard > 60min",
        "Prct retard pour causes externes",
        "Prct retard pour cause infrastructure",
        "Prct retard pour cause gestion trafic",
        "Prct retard pour cause matériel roulant",
        "Prct retard pour cause gestion en gare et réutilisation de matériel",
        "Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)",
        "cancellation_rate",
    ]
    existing = [c for c in numeric_cols if c in df.columns]
    route_avg = (
        df.groupby(["Gare de départ", "Gare d'arrivée"])[existing].mean().reset_index()
    )
    return route_avg


# ── GLOBAL AVERAGES (fallback when route not found) ───────────────────────────
@st.cache_data
def compute_global_averages(df):
    numeric_cols = [
        "Durée moyenne du trajet",
        "Nombre de circulations prévues",
        "Nombre de trains annulés",
        "Nombre de trains en retard au départ",
        "Retard moyen des trains en retard au départ",
        "Retard moyen de tous les trains au départ",
        "Nombre trains en retard > 15min",
        "Nombre trains en retard > 30min",
        "Nombre trains en retard > 60min",
        "Prct retard pour causes externes",
        "Prct retard pour cause infrastructure",
        "Prct retard pour cause gestion trafic",
        "Prct retard pour cause matériel roulant",
        "Prct retard pour cause gestion en gare et réutilisation de matériel",
        "Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)",
        "cancellation_rate",
    ]
    existing = [c for c in numeric_cols if c in df.columns]
    return df[existing].mean()


# ── LOAD EVERYTHING ───────────────────────────────────────────────────────────
df = load_data()
model, feature_cols = load_model()
route_avg = compute_route_averages(df)
global_avg = compute_global_averages(df)

TARGET = "Retard moyen de tous les trains à l'arrivée"

# Station lists
stations_dep = sorted(df["Gare de départ"].dropna().unique().tolist())
stations_arr = sorted(df["Gare d'arrivée"].dropna().unique().tolist())
services = sorted(df["Service"].dropna().unique().tolist())

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/SNCF_logo_2011.svg/200px-SNCF_logo_2011.svg.png",
    width=100,
)
st.sidebar.title("🚄 TARDIS")
st.sidebar.caption("SNCF Delay Prediction System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Explore Data", "🤖 Predict Delay", "📈 Model Performance"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: **{len(df):,}** records  \n"
    f"Routes: **{df['Gare de départ'].nunique()}** stations  \n"
    f"Period: **2018 – 2024**"
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 TARDIS — SNCF Delay Dashboard")
    st.caption("Historical analysis of train delays across France")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    avg_delay = df[TARGET].mean()
    total_records = len(df)
    n_stations = df["Gare de départ"].nunique()
    punctuality = (df[TARGET] < 5).mean() * 100

    col1.metric("🗂️ Total Records", f"{total_records:,}")
    col2.metric("🚉 Unique Stations", str(n_stations))
    col3.metric("⏱️ Avg Arrival Delay", f"{avg_delay:.1f} min")
    col4.metric("✅ Punctuality Rate", f"{punctuality:.1f}%", help="% of routes with avg delay < 5 min")

    st.markdown("---")

    # Chart 1 — Delay over time
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📅 Average Delay Per Year")
        if "year" in df.columns:
            yearly = (
                df.groupby("year")[TARGET]
                .mean()
                .reset_index()
                .rename(columns={"year": "Year", TARGET: "Avg Delay (min)"})
            )
            fig1, ax1 = plt.subplots(figsize=(7, 3.5))
            fig1.patch.set_facecolor("#0e1117")
            ax1.set_facecolor("#0e1117")
            ax1.plot(
                yearly["Year"],
                yearly["Avg Delay (min)"],
                marker="o",
                color="#3b82f6",
                linewidth=2.5,
                markersize=7,
            )
            ax1.fill_between(
                yearly["Year"],
                yearly["Avg Delay (min)"],
                alpha=0.15,
                color="#3b82f6",
            )
            ax1.set_xlabel("Year", color="#94a3b8")
            ax1.set_ylabel("Avg Delay (min)", color="#94a3b8")
            ax1.tick_params(colors="#94a3b8")
            ax1.spines[:].set_color("#1e2d45")
            ax1.set_xticks(yearly["Year"].astype(int))
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()

    with col_b:
        st.subheader("🥧 Delay Causes Breakdown")
        cause_cols = {
            "External": "Prct retard pour causes externes",
            "Infrastructure": "Prct retard pour cause infrastructure",
            "Traffic Mgmt": "Prct retard pour cause gestion trafic",
            "Rolling Stock": "Prct retard pour cause matériel roulant",
            "Station Mgmt": "Prct retard pour cause gestion en gare et réutilisation de matériel",
            "Passengers": "Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)",
        }
        cause_means = {
            k: df[v].mean() for k, v in cause_cols.items() if v in df.columns
        }
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#0e1117")
        colors = ["#3b82f6", "#06b6d4", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444"]
        wedges, texts, autotexts = ax2.pie(
            cause_means.values(),
            labels=cause_means.keys(),
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            textprops={"color": "#94a3b8", "fontsize": 8},
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(7)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Chart 2 — Delay distribution
    st.subheader("📈 Distribution of Average Arrival Delays")
    fig3, ax3 = plt.subplots(figsize=(12, 3.5))
    fig3.patch.set_facecolor("#0e1117")
    ax3.set_facecolor("#0e1117")
    sns.histplot(
        df[TARGET].dropna(),
        bins=60,
        color="#3b82f6",
        kde=True,
        ax=ax3,
        alpha=0.7,
    )
    ax3.set_xlabel("Average Delay (minutes)", color="#94a3b8")
    ax3.set_ylabel("Count", color="#94a3b8")
    ax3.tick_params(colors="#94a3b8")
    ax3.spines[:].set_color("#1e2d45")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    st.caption(
        "Most routes have an average delay between 2–10 minutes. "
        "The right-skewed distribution shows that very high delays are rare but do occur."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORE DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Explore Data":
    st.title("🔍 Explore Train Delay Data")
    st.caption("Filter by year, station, or service type")
    st.markdown("---")

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        years = ["All"] + sorted(df["year"].dropna().astype(int).unique().tolist()) if "year" in df.columns else ["All"]
        sel_year = st.selectbox("📅 Year", years)
    with f2:
        sel_station = st.selectbox("🚉 Departure Station", ["All"] + stations_dep)
    with f3:
        sel_service = st.selectbox("🚆 Service Type", ["All"] + services)

    # Apply filters
    filtered = df.copy()
    if sel_year != "All":
        filtered = filtered[filtered["year"] == int(sel_year)]
    if sel_station != "All":
        filtered = filtered[filtered["Gare de départ"] == sel_station]
    if sel_service != "All":
        filtered = filtered[filtered["Service"] == sel_service]

    st.info(f"Showing **{len(filtered):,}** records matching your filters.")

    # Top 10 delayed stations
    st.subheader("🏆 Top 10 Most Delayed Departure Stations")
    top_delayed = (
        filtered.groupby("Gare de départ")[TARGET]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig4, ax4 = plt.subplots(figsize=(12, 4.5))
    fig4.patch.set_facecolor("#0e1117")
    ax4.set_facecolor("#0e1117")
    colors_bar = [
        f"#{int(255 - i * 18):02x}{int(50 + i * 8):02x}{int(50):02x}"
        for i in range(len(top_delayed))
    ]
    ax4.barh(
        top_delayed["Gare de départ"][::-1],
        top_delayed[TARGET][::-1],
        color=colors_bar[::-1],
    )
    ax4.set_xlabel("Average Delay (minutes)", color="#94a3b8")
    ax4.tick_params(colors="#94a3b8")
    ax4.spines[:].set_color("#1e2d45")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # Monthly trend
    if "month" in filtered.columns and len(filtered) > 0:
        st.subheader("📆 Average Delay by Month")
        monthly = (
            filtered.groupby("month")[TARGET]
            .mean()
            .reset_index()
            .rename(columns={"month": "Month"})
        )
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
            5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        monthly["Month Name"] = monthly["Month"].map(month_names)
        fig5, ax5 = plt.subplots(figsize=(12, 3.5))
        fig5.patch.set_facecolor("#0e1117")
        ax5.set_facecolor("#0e1117")
        ax5.bar(
            monthly["Month Name"],
            monthly[TARGET],
            color="#3b82f6",
            alpha=0.8,
        )
        ax5.set_ylabel("Avg Delay (min)", color="#94a3b8")
        ax5.tick_params(colors="#94a3b8")
        ax5.spines[:].set_color("#1e2d45")
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    # Raw data table
    st.subheader("📋 Data Table")
    display_cols = [
        "Gare de départ",
        "Gare d'arrivée",
        "Service",
        TARGET,
        "Nombre de circulations prévues",
        "Nombre de trains annulés",
    ]
    existing_display = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[existing_display]
        .sort_values(TARGET, ascending=False)
        .head(100)
        .reset_index(drop=True),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT DELAY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict Delay":
    st.title("🤖 Predict Train Delay")
    st.caption(
        "Enter your journey details below. The model uses historical patterns "
        "from similar routes to estimate the expected arrival delay."
    )
    st.markdown("---")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("🚄 Journey Details")

        dep = st.selectbox("Departure Station", stations_dep, index=stations_dep.index("PARIS MONTPARNASSE") if "PARIS MONTPARNASSE" in stations_dep else 0)
        arr = st.selectbox("Arrival Station", stations_arr, index=stations_arr.index("BORDEAUX SAINT JEAN") if "BORDEAUX SAINT JEAN" in stations_arr else 1)
        service = st.selectbox("Service Type", services)

        col_y, col_m = st.columns(2)
        with col_y:
            year = st.selectbox("Year", list(range(2024, 2031)), index=0)
        with col_m:
            month_names = {
                "January": 1, "February": 2, "March": 3, "April": 4,
                "May": 5, "June": 6, "July": 7, "August": 8,
                "September": 9, "October": 10, "November": 11, "December": 12,
            }
            month_name = st.selectbox("Month", list(month_names.keys()))
            month = month_names[month_name]

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Delay", type="primary", use_container_width=True)

    with col_result:
        st.subheader("📍 Prediction Result")

        if predict_btn:
            # Get historical averages for this route
            route_data = route_avg[
                (route_avg["Gare de départ"] == dep) &
                (route_avg["Gare d'arrivée"] == arr)
            ]

            if len(route_data) > 0:
                row = route_data.iloc[0]
                source = "route historical average"
            else:
                row = global_avg
                source = "global average (new route)"

            # Build input row with get_dummies columns
            input_dict = {}

            # Numeric features from historical averages
            numeric_features = [
                "Durée moyenne du trajet",
                "Nombre de circulations prévues",
                "Nombre de trains annulés",
                "Nombre de trains en retard au départ",
                "Retard moyen des trains en retard au départ",
                "Retard moyen de tous les trains au départ",
                "Nombre trains en retard > 15min",
                "Nombre trains en retard > 30min",
                "Nombre trains en retard > 60min",
                "Prct retard pour causes externes",
                "Prct retard pour cause infrastructure",
                "Prct retard pour cause gestion trafic",
                "Prct retard pour cause matériel roulant",
                "Prct retard pour cause gestion en gare et réutilisation de matériel",
                "Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)",
                "cancellation_rate",
            ]

            for col in numeric_features:
                if col in row.index:
                    input_dict[col] = float(row[col])
                elif col in global_avg.index:
                    input_dict[col] = float(global_avg[col])
                else:
                    input_dict[col] = 0.0

            # Year and month from user input
            input_dict["year"] = year
            input_dict["month"] = month

            # Build full feature vector matching model_features columns
            input_row = pd.DataFrame([{col: 0 for col in feature_cols}])

            # Fill numeric features
            for col, val in input_dict.items():
                if col in input_row.columns:
                    input_row[col] = val

            # One-hot encode service
            service_col = f"Service_{service}"
            if service_col in input_row.columns:
                input_row[service_col] = 1

            # One-hot encode departure station
            dep_col = f"Gare de départ_{dep}"
            if dep_col in input_row.columns:
                input_row[dep_col] = 1

            # One-hot encode arrival station
            arr_col = f"Gare d'arrivée_{arr}"
            if arr_col in input_row.columns:
                input_row[arr_col] = 1

            # Predict
            prediction = float(model.predict(input_row)[0])
            prediction = max(0.0, prediction)

            # Display result
            if prediction < 5:
                status = "✅ On Time"
                color = "green"
                badge = "success"
            elif prediction < 10:
                status = "⚠️ Minor Delay"
                color = "orange"
                badge = "warning"
            else:
                status = "🚨 Significant Delay"
                color = "red"
                badge = "error"

            # Big result display
            st.markdown(
                f"""
                <div style='background:#1e2535;border-radius:12px;padding:2rem;text-align:center;border:1px solid #2d3748;margin-bottom:1rem'>
                    <p style='color:#94a3b8;font-size:0.8rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem'>
                        Estimated Arrival Delay
                    </p>
                    <p style='color:{color};font-size:3.5rem;font-weight:700;line-height:1;margin:0'>
                        {prediction:.1f}
                    </p>
                    <p style='color:#94a3b8;font-size:1rem;margin-top:0.3rem'>minutes</p>
                    <p style='color:{color};font-size:1rem;font-weight:600;margin-top:0.8rem'>{status}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Journey details
            st.markdown("**Journey Summary**")
            details = {
                "🚉 Route": f"{dep} → {arr}",
                "🚆 Service": service,
                "📅 Period": f"{month_name} {year}",
                "📊 Data source": source,
                "🤖 Model": "Random Forest (R² = 0.81)",
            }
            for k, v in details.items():
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:0.4rem 0.6rem;background:#1e2535;border-radius:6px;"
                    f"margin-bottom:4px'><span style='color:#64748b'>{k}</span>"
                    f"<span style='color:#e2e8f0;font-weight:500'>{v}</span></div>",
                    unsafe_allow_html=True,
                )

            # Show route historical stats if available
            if len(route_data) > 0 and "Durée moyenne du trajet" in route_data.columns:
                st.markdown("**Historical Route Stats**")
                avg_dur = route_data["Durée moyenne du trajet"].values[0]
                avg_hist_delay = df[
                    (df["Gare de départ"] == dep) & (df["Gare d'arrivée"] == arr)
                ][TARGET].mean()
                c1, c2 = st.columns(2)
                c1.metric("Avg Trip Duration", f"{avg_dur:.0f} min")
                c2.metric("Historical Avg Delay", f"{avg_hist_delay:.1f} min")

        else:
            st.markdown(
                """
                <div style='background:#1e2535;border-radius:12px;padding:2.5rem;
                text-align:center;border:1px dashed #2d3748'>
                    <p style='font-size:3rem;margin:0'>🚄</p>
                    <p style='color:#64748b;margin-top:0.5rem'>
                        Fill in your journey details<br>and click <strong>Predict Delay</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(
                "💡 The model uses historical patterns from similar routes to estimate "
                "the expected arrival delay for your future journey."
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.caption("Comparison of Baseline, Linear Regression and Random Forest")
    st.markdown("---")

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 RMSE", "1.52 min", help="Average prediction error in minutes — lower is better")
    c2.metric("📏 MAE", "1.05 min", help="Mean absolute error — lower is better")
    c3.metric("📊 R² Score", "0.81", help="Variance explained by model — higher is better (max 1.0)")

    st.markdown("---")

    # Model comparison chart
    st.subheader("🏆 Model Comparison — RMSE (lower is better)")
    comparison = pd.DataFrame({
        "Model": ["Baseline (mean)", "Linear Regression", "Random Forest"],
        "RMSE": [3.51, 3.10, 1.52],
    })
    fig6, ax6 = plt.subplots(figsize=(10, 3))
    fig6.patch.set_facecolor("#0e1117")
    ax6.set_facecolor("#0e1117")
    bar_colors = ["#ef4444", "#f59e0b", "#10b981"]
    bars = ax6.barh(comparison["Model"], comparison["RMSE"], color=bar_colors, height=0.5)
    for bar, val in zip(bars, comparison["RMSE"]):
        ax6.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{val} min",
            va="center",
            color="#e2e8f0",
            fontsize=11,
            fontweight="bold",
        )
    ax6.set_xlabel("RMSE (minutes)", color="#94a3b8")
    ax6.tick_params(colors="#94a3b8")
    ax6.spines[:].set_color("#1e2d45")
    ax6.set_xlim(0, 4.5)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

    st.caption(
        "Random Forest is **2.3x more accurate** than the baseline. "
        "Linear Regression barely improves over the baseline, confirming that "
        "delay patterns are non-linear and require a tree-based model."
    )

    st.markdown("---")

    # Full comparison table
    st.subheader("📋 Full Metrics Table")
    metrics_df = pd.DataFrame({
        "Model": ["Baseline (mean)", "Linear Regression", "Random Forest ✓"],
        "RMSE (min)": [3.51, 3.10, 1.52],
        "MAE (min)": ["—", 2.34, 1.05],
        "R²": ["—", 0.2231, 0.8122],
        "Beats Baseline": ["—", "✅ Yes", "✅ Yes (2.3x)"],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Method explanation
    st.subheader("🧠 About the Model")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            **Why Random Forest?**
            - Handles non-linear relationships between features
            - Robust to outliers and noisy data
            - No need for feature scaling
            - Built-in feature importance
            - 200 trees averaged = stable predictions
            """
        )

    with col_b:
        st.markdown(
            """
            **Training Details**
            - Algorithm: Random Forest Regressor
            - Trees: 200 (`n_estimators=200`)
            - Min samples per leaf: 2
            - Encoding: One-hot (get_dummies)
            - Train/test split: 80% / 20%
            - Outlier removal: bottom & top 1%
            """
        )