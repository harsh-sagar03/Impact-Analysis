from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "naye_pankh_sample_impact_data.csv"
MODELING_PATH = ROOT / "data" / "modeling_dataset.csv"
CONTEXT_PATH = ROOT / "data" / "india_state_context.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
FORECAST_PATH = ROOT / "outputs" / "forecast_vs_actual.csv"
DONATION_MODEL_PATH = ROOT / "models" / "saved" / "donation_model.joblib"
BENEFICIARY_MODEL_PATH = ROOT / "models" / "saved" / "beneficiary_model.joblib"
EFFICIENCY_MODEL_PATH = ROOT / "models" / "saved" / "efficiency_model.joblib"

ACCENT_COLORS = ["#0F9F8F", "#F9734A", "#4F46E5", "#D99A00", "#2563EB", "#DB2777"]


st.set_page_config(
    page_title="NayePankh Impact Analytics",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.95rem 1rem;
        background: #ffffff;
        min-height: 116px;
    }
    .metric-label {
        color: #667085;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        color: #111827;
        font-size: 1.65rem;
        line-height: 1.2;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .metric-note {
        color: #667085;
        font-size: 0.84rem;
    }
    .insight-box {
        border-left: 4px solid #0F9F8F;
        background: #F8FAFC;
        color: #111827;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.75rem;
    }
    .small-muted {
        color: #667085;
        font-size: 0.88rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem;
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["month_label"] = df["date"].dt.strftime("%b %Y")
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df["cost_per_beneficiary"] = df["expense_amount"] / df["beneficiaries_reached"]
    df["beneficiaries_per_volunteer_hour"] = (
        df["beneficiaries_reached"] / df["volunteer_hours"]
    )
    df["donation_surplus"] = df["donation_amount"] - df["expense_amount"]
    return df


@st.cache_data
def load_modeling_data() -> pd.DataFrame:
    if not MODELING_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MODELING_PATH, parse_dates=["date"])


@st.cache_data
def load_context_data() -> pd.DataFrame:
    if not CONTEXT_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CONTEXT_PATH)


@st.cache_resource
def load_saved_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def load_model_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_scenario_input(
    modeling_df: pd.DataFrame,
    city: str,
    program: str,
    donor_type: str,
    channel: str,
    month_num: int,
    volunteer_hours: float,
    expense_amount: float,
    event_count: int,
    social_reach: float,
    engagement_rate: float,
) -> pd.DataFrame:
    history = modeling_df[
        (modeling_df["city"] == city) & (modeling_df["program"] == program)
    ].sort_values("date")

    if history.empty:
        raise ValueError("No historical records found for the selected city and program.")

    latest = history.iloc[-1]
    is_festive = int(month_num in {8, 9, 10, 11, 12})
    state_row = modeling_df[modeling_df["state"] == latest["state"]].iloc[0]

    payload = {
        "month_num": month_num,
        "is_festive_season": is_festive,
        "city": city,
        "program": program,
        "donor_type": donor_type,
        "channel": channel,
        "volunteer_hours": volunteer_hours,
        "event_count": event_count,
        "social_reach": social_reach,
        "engagement_rate": engagement_rate,
        "expense_amount": expense_amount,
        "donation_lag_1": latest["donation_amount"],
        "donation_lag_2": history.iloc[-2]["donation_amount"] if len(history) > 1 else latest["donation_amount"],
        "donation_lag_3": history.iloc[-3]["donation_amount"] if len(history) > 2 else latest["donation_amount"],
        "beneficiaries_lag_1": latest["beneficiaries_reached"],
        "beneficiaries_lag_2": history.iloc[-2]["beneficiaries_reached"] if len(history) > 1 else latest["beneficiaries_reached"],
        "literacy_rate_pct": state_row["literacy_rate_pct"],
        "multidimensional_poverty_index": state_row["multidimensional_poverty_index"],
    }
    return pd.DataFrame([payload])


def money(value: float) -> str:
    if abs(value) >= 10000000:
        return f"₹{value / 10000000:.2f} Cr"
    if abs(value) >= 100000:
        return f"₹{value / 100000:.2f} L"
    return f"₹{value:,.0f}"


def compact_number(value: float) -> str:
    if abs(value) >= 100000:
        return f"{value / 100000:.2f} L"
    if abs(value) >= 1000:
        return f"{value / 1000:.1f}K"
    return f"{value:,.0f}"


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def grouped_bar(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    return px.bar(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        color_discrete_sequence=ACCENT_COLORS,
        text_auto=".2s",
    ).update_layout(
        legend_title_text="",
        margin=dict(l=20, r=20, t=55, b=20),
        yaxis_title="",
        xaxis_title="",
    )


def build_recommendations(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No records match the selected filters."]

    program_summary = (
        df.groupby("program")
        .agg(
            beneficiaries=("beneficiaries_reached", "sum"),
            donation=("donation_amount", "sum"),
            expense=("expense_amount", "sum"),
            volunteer_hours=("volunteer_hours", "sum"),
        )
        .assign(
            cost_per_beneficiary=lambda x: x["expense"] / x["beneficiaries"],
            beneficiaries_per_hour=lambda x: x["beneficiaries"] / x["volunteer_hours"],
        )
    )
    city_summary = (
        df.groupby("city")
        .agg(
            beneficiaries=("beneficiaries_reached", "sum"),
            volunteer_hours=("volunteer_hours", "sum"),
            donation=("donation_amount", "sum"),
        )
        .assign(hours_per_100_beneficiaries=lambda x: x["volunteer_hours"] / x["beneficiaries"] * 100)
    )
    campaign_summary = (
        df.groupby("campaign")
        .agg(donation=("donation_amount", "sum"), beneficiaries=("beneficiaries_reached", "sum"))
        .assign(donation_per_beneficiary=lambda x: x["donation"] / x["beneficiaries"])
    )

    best_efficiency = program_summary["cost_per_beneficiary"].idxmin()
    highest_reach = program_summary["beneficiaries"].idxmax()
    volunteer_gap_city = city_summary["hours_per_100_beneficiaries"].idxmin()
    strongest_campaign = campaign_summary["donation"].idxmax()
    underleveraged_campaign = campaign_summary["donation_per_beneficiary"].idxmin()

    return [
        f"{best_efficiency} shows the lowest cost per beneficiary at {money(program_summary.loc[best_efficiency, 'cost_per_beneficiary'])}, making it a strong model for efficient delivery.",
        f"{highest_reach} has reached the most beneficiaries, so it should be reviewed for repeatable outreach practices.",
        f"{volunteer_gap_city} has the lowest volunteer support relative to beneficiary volume, indicating a possible staffing or volunteer allocation gap.",
        f"{strongest_campaign} generated the highest donation total and can be used as the benchmark for campaign messaging.",
        f"{underleveraged_campaign} is reaching beneficiaries with comparatively lower funding per person, making it a candidate for targeted donor storytelling.",
    ]


def build_report(df: pd.DataFrame) -> str:
    total_donation = df["donation_amount"].sum()
    total_expense = df["expense_amount"].sum()
    total_beneficiaries = df["beneficiaries_reached"].sum()
    total_hours = df["volunteer_hours"].sum()
    cost_per_beneficiary = total_expense / total_beneficiaries
    top_city = df.groupby("city")["beneficiaries_reached"].sum().idxmax()
    top_campaign = df.groupby("campaign")["donation_amount"].sum().idxmax()
    top_program = df.groupby("program")["beneficiaries_reached"].sum().idxmax()

    recommendations = "\n".join(f"- {item}" for item in build_recommendations(df))
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"

    return f"""# NayePankh Foundation Impact Analytics Summary

Dataset period: {date_range}

This automated summary is based on a simulated dataset created for a Data Analytics internship project. It demonstrates how a foundation can monitor fundraising, social impact, volunteer effort, and campaign efficiency through a single reporting view.

## Key Performance Indicators

- Total donations tracked: {money(total_donation)}
- Total expenses tracked: {money(total_expense)}
- Beneficiaries reached: {compact_number(total_beneficiaries)}
- Volunteer hours contributed: {compact_number(total_hours)}
- Average cost per beneficiary: {money(cost_per_beneficiary)}

## Main Findings

- {top_campaign} was the strongest campaign by donation value.
- {top_program} created the highest beneficiary reach.
- {top_city} recorded the highest city-level impact.
- The overall surplus across the selected data is {money(total_donation - total_expense)}.

## Recommendations

{recommendations}
"""


df = load_data()

if df.empty:
    st.error("Sample dataset not found. Run `python scripts/generate_sample_data.py` from the project folder.")
    st.stop()

st.title("NayePankh Foundation Impact Intelligence Platform")
st.caption(
    "Predictive analytics and decision dashboard built for a Data Analytics internship project. "
    "Operational data is simulated; state context data is from open public indicators."
)

with st.sidebar:
    st.header("Filters")
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    selected_cities = st.multiselect(
        "City",
        options=sorted(df["city"].unique()),
        default=sorted(df["city"].unique()),
    )
    selected_programs = st.multiselect(
        "Program",
        options=sorted(df["program"].unique()),
        default=sorted(df["program"].unique()),
    )
    selected_donors = st.multiselect(
        "Donor type",
        options=sorted(df["donor_type"].unique()),
        default=sorted(df["donor_type"].unique()),
    )

if len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date, end_date = min_date, max_date

filtered = df[
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
    & (df["city"].isin(selected_cities))
    & (df["program"].isin(selected_programs))
    & (df["donor_type"].isin(selected_donors))
].copy()

if filtered.empty:
    st.warning("No records match the current filter selection.")
    st.stop()

total_donation = filtered["donation_amount"].sum()
total_expense = filtered["expense_amount"].sum()
total_beneficiaries = filtered["beneficiaries_reached"].sum()
total_hours = filtered["volunteer_hours"].sum()
cost_per_beneficiary = total_expense / total_beneficiaries
beneficiaries_per_hour = total_beneficiaries / total_hours
surplus = total_donation - total_expense

monthly = (
    filtered.groupby(pd.Grouper(key="date", freq="MS"))
    .agg(
        donation_amount=("donation_amount", "sum"),
        expense_amount=("expense_amount", "sum"),
        beneficiaries_reached=("beneficiaries_reached", "sum"),
        volunteer_hours=("volunteer_hours", "sum"),
        event_count=("event_count", "sum"),
        social_reach=("social_reach", "sum"),
    )
    .reset_index()
)
monthly["month_label"] = monthly["date"].dt.strftime("%b %Y")

previous_period = monthly.tail(2)
if len(previous_period) == 2 and previous_period.iloc[0]["donation_amount"] != 0:
    donation_delta = (
        (previous_period.iloc[1]["donation_amount"] - previous_period.iloc[0]["donation_amount"])
        / previous_period.iloc[0]["donation_amount"]
        * 100
    )
else:
    donation_delta = 0

cols = st.columns(6)
with cols[0]:
    metric_card("Donations", money(total_donation), f"{donation_delta:+.1f}% latest month")
with cols[1]:
    metric_card("Beneficiaries", compact_number(total_beneficiaries), "People reached")
with cols[2]:
    metric_card("Volunteer Hours", compact_number(total_hours), "Community effort")
with cols[3]:
    metric_card("Cost / Beneficiary", money(cost_per_beneficiary), "Delivery efficiency")
with cols[4]:
    metric_card("Beneficiaries / Hour", f"{beneficiaries_per_hour:.1f}", "Volunteer productivity")
with cols[5]:
    metric_card("Net Position", money(surplus), "Donations minus expenses")

overview_tab, donations_tab, impact_tab, volunteers_tab, predictions_tab, context_tab, report_tab = st.tabs(
    ["Overview", "Donations", "Impact", "Volunteer Ops", "Predictions", "Geographic Context", "Auto Report"]
)

with overview_tab:
    left, right = st.columns([1.45, 1])

    with left:
        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                x=monthly["date"],
                y=monthly["donation_amount"],
                mode="lines+markers",
                name="Donations",
                line=dict(color="#0F9F8F", width=3),
            )
        )
        trend_fig.add_trace(
            go.Scatter(
                x=monthly["date"],
                y=monthly["expense_amount"],
                mode="lines+markers",
                name="Expenses",
                line=dict(color="#F9734A", width=3),
            )
        )
        trend_fig.update_layout(
            title="Monthly Donations vs Expenses",
            xaxis_title="",
            yaxis_title="Amount",
            margin=dict(l=20, r=20, t=55, b=20),
            legend_title_text="",
        )
        st.plotly_chart(trend_fig, use_container_width=True)

    with right:
        st.subheader("Decision Notes")
        for insight in build_recommendations(filtered)[:4]:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    program_rollup = (
        filtered.groupby("program")
        .agg(
            donation_amount=("donation_amount", "sum"),
            beneficiaries_reached=("beneficiaries_reached", "sum"),
            expense_amount=("expense_amount", "sum"),
            volunteer_hours=("volunteer_hours", "sum"),
        )
        .reset_index()
    )
    program_rollup["cost_per_beneficiary"] = (
        program_rollup["expense_amount"] / program_rollup["beneficiaries_reached"]
    )

    st.plotly_chart(
        px.scatter(
            program_rollup,
            x="donation_amount",
            y="beneficiaries_reached",
            size="volunteer_hours",
            color="program",
            color_discrete_sequence=ACCENT_COLORS,
            title="Program Positioning: Funding, Reach, and Volunteer Effort",
            hover_data={
                "donation_amount": ":,.0f",
                "beneficiaries_reached": ":,.0f",
                "volunteer_hours": ":,.1f",
                "program": False,
            },
        ).update_layout(
            xaxis_title="Donations",
            yaxis_title="Beneficiaries Reached",
            legend_title_text="",
            margin=dict(l=20, r=20, t=55, b=20),
        ),
        use_container_width=True,
    )

with donations_tab:
    campaign_rollup = (
        filtered.groupby(["campaign", "program"])
        .agg(
            donation_amount=("donation_amount", "sum"),
            expense_amount=("expense_amount", "sum"),
            new_donors=("new_donors", "sum"),
            recurring_donors=("recurring_donors", "sum"),
        )
        .reset_index()
        .sort_values("donation_amount", ascending=False)
    )
    donor_rollup = (
        filtered.groupby("donor_type")
        .agg(donation_amount=("donation_amount", "sum"), new_donors=("new_donors", "sum"))
        .reset_index()
        .sort_values("donation_amount", ascending=False)
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            grouped_bar(
                campaign_rollup,
                "campaign",
                "donation_amount",
                "program",
                "Donation Raised by Campaign",
            ),
            use_container_width=True,
        )
    with col_b:
        donor_fig = px.pie(
            donor_rollup,
            names="donor_type",
            values="donation_amount",
            title="Donation Share by Donor Type",
            color_discrete_sequence=ACCENT_COLORS,
            hole=0.48,
        ).update_layout(margin=dict(l=20, r=20, t=55, b=20), legend_title_text="")
        st.plotly_chart(donor_fig, use_container_width=True)

    campaign_rollup["funding_coverage"] = (
        campaign_rollup["donation_amount"] / campaign_rollup["expense_amount"]
    )
    st.dataframe(
        campaign_rollup.assign(
            donation_amount=campaign_rollup["donation_amount"].map(money),
            expense_amount=campaign_rollup["expense_amount"].map(money),
            funding_coverage=campaign_rollup["funding_coverage"].map(lambda x: f"{x:.2f}x"),
        ),
        width="stretch",
        hide_index=True,
    )

with impact_tab:
    city_rollup = (
        filtered.groupby(["city", "state"])
        .agg(
            beneficiaries_reached=("beneficiaries_reached", "sum"),
            expense_amount=("expense_amount", "sum"),
            volunteer_hours=("volunteer_hours", "sum"),
            social_reach=("social_reach", "sum"),
        )
        .reset_index()
    )
    city_rollup["cost_per_beneficiary"] = (
        city_rollup["expense_amount"] / city_rollup["beneficiaries_reached"]
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            px.bar(
                city_rollup.sort_values("beneficiaries_reached", ascending=True),
                x="beneficiaries_reached",
                y="city",
                color="cost_per_beneficiary",
                orientation="h",
                title="City-wise Beneficiary Reach and Delivery Cost",
                color_continuous_scale=["#D1FAE5", "#FBBF24", "#F9734A"],
                text_auto=".2s",
            ).update_layout(
                xaxis_title="Beneficiaries Reached",
                yaxis_title="",
                margin=dict(l=20, r=20, t=55, b=20),
                coloraxis_colorbar_title="Cost / Beneficiary",
            ),
            use_container_width=True,
        )
    with col_b:
        impact_trend = go.Figure()
        impact_trend.add_trace(
            go.Bar(
                x=monthly["date"],
                y=monthly["beneficiaries_reached"],
                name="Beneficiaries",
                marker_color="#4F46E5",
            )
        )
        impact_trend.add_trace(
            go.Scatter(
                x=monthly["date"],
                y=monthly["social_reach"],
                name="Social Reach",
                yaxis="y2",
                line=dict(color="#D99A00", width=3),
            )
        )
        impact_trend.update_layout(
            title="Impact Reach Trend",
            xaxis_title="",
            yaxis_title="Beneficiaries",
            yaxis2=dict(title="Social Reach", overlaying="y", side="right"),
            legend_title_text="",
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(impact_trend, use_container_width=True)

    program_efficiency = (
        filtered.groupby("program")
        .agg(
            beneficiaries_reached=("beneficiaries_reached", "sum"),
            expense_amount=("expense_amount", "sum"),
            volunteer_hours=("volunteer_hours", "sum"),
            event_count=("event_count", "sum"),
        )
        .reset_index()
    )
    program_efficiency["cost_per_beneficiary"] = (
        program_efficiency["expense_amount"] / program_efficiency["beneficiaries_reached"]
    )
    program_efficiency["beneficiaries_per_event"] = (
        program_efficiency["beneficiaries_reached"] / program_efficiency["event_count"]
    )
    st.plotly_chart(
        px.bar(
            program_efficiency.sort_values("cost_per_beneficiary"),
            x="program",
            y="cost_per_beneficiary",
            color="program",
            title="Cost per Beneficiary by Program",
            color_discrete_sequence=ACCENT_COLORS,
            text_auto=".2f",
        ).update_layout(
            xaxis_title="",
            yaxis_title="Cost per Beneficiary",
            showlegend=False,
            margin=dict(l=20, r=20, t=55, b=20),
        ),
        use_container_width=True,
    )

with volunteers_tab:
    volunteer_rollup = (
        filtered.groupby(["city", "program"])
        .agg(
            volunteer_hours=("volunteer_hours", "sum"),
            beneficiaries_reached=("beneficiaries_reached", "sum"),
            event_count=("event_count", "sum"),
        )
        .reset_index()
    )
    volunteer_rollup["beneficiaries_per_hour"] = (
        volunteer_rollup["beneficiaries_reached"] / volunteer_rollup["volunteer_hours"]
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            px.bar(
                volunteer_rollup,
                x="city",
                y="volunteer_hours",
                color="program",
                title="Volunteer Hours by City and Program",
                color_discrete_sequence=ACCENT_COLORS,
            ).update_layout(
                xaxis_title="",
                yaxis_title="Volunteer Hours",
                legend_title_text="",
                margin=dict(l=20, r=20, t=55, b=20),
            ),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            px.scatter(
                volunteer_rollup,
                x="volunteer_hours",
                y="beneficiaries_reached",
                color="program",
                size="event_count",
                hover_name="city",
                title="Volunteer Productivity by Location",
                color_discrete_sequence=ACCENT_COLORS,
            ).update_layout(
                xaxis_title="Volunteer Hours",
                yaxis_title="Beneficiaries Reached",
                legend_title_text="",
                margin=dict(l=20, r=20, t=55, b=20),
            ),
            use_container_width=True,
        )

    ranking = volunteer_rollup.sort_values("beneficiaries_per_hour", ascending=False)
    st.dataframe(
        ranking.assign(
            volunteer_hours=ranking["volunteer_hours"].map(lambda x: f"{x:,.1f}"),
            beneficiaries_per_hour=ranking["beneficiaries_per_hour"].map(lambda x: f"{x:.2f}"),
        ),
        width="stretch",
        hide_index=True,
    )

with predictions_tab:
    metrics_payload = load_model_metrics()
    modeling_df = load_modeling_data()
    donation_model = load_saved_model(DONATION_MODEL_PATH)
    beneficiary_model = load_saved_model(BENEFICIARY_MODEL_PATH)
    efficiency_model = load_saved_model(EFFICIENCY_MODEL_PATH)

    if not metrics_payload or modeling_df.empty or donation_model is None:
        st.info(
            "Run the modeling pipeline first:\n\n"
            "`python scripts/generate_sample_data.py`\n\n"
            "`python scripts/prepare_data.py`\n\n"
            "`python scripts/train_models.py`"
        )
    else:
        st.subheader("Model Comparison")
        results_df = pd.DataFrame(metrics_payload["results"])
        display_rows = []
        for _, row in results_df.iterrows():
            metric_text = ", ".join(f"{key.upper()}: {value}" for key, value in row["metrics"].items())
            display_rows.append(
                {
                    "Task": row["task"],
                    "Model": row["model_name"],
                    "Metrics": metric_text,
                    "Recommended": "Yes" if row["recommended"] else "",
                }
            )
        st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)

        recommended = metrics_payload.get("recommended_models", {})
        if recommended:
            st.markdown(
                " ".join(
                    f'<span class="insight-box"><strong>{task.replace("_", " ").title()}:</strong> {model}</span>'
                    for task, model in recommended.items()
                ),
                unsafe_allow_html=True,
            )

        col_a, col_b = st.columns(2)
        donation_importance_path = ROOT / "outputs" / "donation_feature_importance.csv"
        beneficiary_importance_path = ROOT / "outputs" / "beneficiary_feature_importance.csv"

        with col_a:
            if donation_importance_path.exists():
                importance_df = pd.read_csv(donation_importance_path)
                st.plotly_chart(
                    px.bar(
                        importance_df.head(10),
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Top Features: Donation Forecasting",
                        color_discrete_sequence=["#0F9F8F"],
                    ).update_layout(margin=dict(l=20, r=20, t=55, b=20), yaxis_title=""),
                    use_container_width=True,
                )

        with col_b:
            if beneficiary_importance_path.exists():
                importance_df = pd.read_csv(beneficiary_importance_path)
                st.plotly_chart(
                    px.bar(
                        importance_df.head(10),
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Top Features: Beneficiary Reach",
                        color_discrete_sequence=["#4F46E5"],
                    ).update_layout(margin=dict(l=20, r=20, t=55, b=20), yaxis_title=""),
                    use_container_width=True,
                )

        if FORECAST_PATH.exists():
            forecast_df = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
            donation_forecast = forecast_df[forecast_df["target"] == "donation_amount"].copy()
            if not donation_forecast.empty:
                monthly_forecast = (
                    donation_forecast.groupby("date", as_index=False)
                    .agg(actual=("actual", "sum"), predicted=("predicted", "sum"))
                )
                forecast_fig = go.Figure()
                forecast_fig.add_trace(
                    go.Scatter(
                        x=monthly_forecast["date"],
                        y=monthly_forecast["actual"],
                        mode="lines+markers",
                        name="Actual Donations",
                        line=dict(color="#0F9F8F", width=3),
                    )
                )
                forecast_fig.add_trace(
                    go.Scatter(
                        x=monthly_forecast["date"],
                        y=monthly_forecast["predicted"],
                        mode="lines+markers",
                        name="Predicted Donations",
                        line=dict(color="#F9734A", width=3, dash="dash"),
                    )
                )
                forecast_fig.update_layout(
                    title="2025 Donation Forecast vs Actual (Best Model)",
                    xaxis_title="",
                    yaxis_title="Donation Amount",
                    margin=dict(l=20, r=20, t=55, b=20),
                )
                st.plotly_chart(forecast_fig, use_container_width=True)

        st.subheader("Scenario Simulator")
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            scenario_city = st.selectbox("City", sorted(modeling_df["city"].unique()), key="scenario_city")
            scenario_program = st.selectbox("Program", sorted(modeling_df["program"].unique()), key="scenario_program")
            scenario_month = st.selectbox("Month", list(range(1, 13)), index=9, key="scenario_month")
        with sim_col2:
            scenario_donor = st.selectbox("Donor type", sorted(modeling_df["donor_type"].unique()), key="scenario_donor")
            scenario_channel = st.selectbox("Channel", sorted(modeling_df["channel"].unique()), key="scenario_channel")
            scenario_hours = st.number_input("Volunteer hours", min_value=10.0, max_value=500.0, value=80.0, step=5.0)
        with sim_col3:
            scenario_expense = st.number_input("Planned expense (₹)", min_value=5000, max_value=250000, value=35000, step=1000)
            scenario_events = st.number_input("Event count", min_value=1, max_value=20, value=3, step=1)
            scenario_social = st.number_input("Expected social reach", min_value=1000, max_value=50000, value=12000, step=500)

        scenario_engagement = float(
            modeling_df["engagement_rate"].mean()
        )

        if st.button("Run Scenario Prediction", type="primary"):
            scenario_input = build_scenario_input(
                modeling_df=modeling_df,
                city=scenario_city,
                program=scenario_program,
                donor_type=scenario_donor,
                channel=scenario_channel,
                month_num=scenario_month,
                volunteer_hours=scenario_hours,
                expense_amount=float(scenario_expense),
                event_count=int(scenario_events),
                social_reach=float(scenario_social),
                engagement_rate=scenario_engagement,
            )

            donation_features = scenario_input[
                [
                    "month_num",
                    "is_festive_season",
                    "city",
                    "program",
                    "donor_type",
                    "channel",
                    "volunteer_hours",
                    "event_count",
                    "social_reach",
                    "engagement_rate",
                    "donation_lag_1",
                    "donation_lag_2",
                    "donation_lag_3",
                    "literacy_rate_pct",
                    "multidimensional_poverty_index",
                ]
            ]
            beneficiary_features = scenario_input[
                [
                    "month_num",
                    "is_festive_season",
                    "city",
                    "program",
                    "expense_amount",
                    "volunteer_hours",
                    "event_count",
                    "social_reach",
                    "engagement_rate",
                    "beneficiaries_lag_1",
                    "beneficiaries_lag_2",
                    "literacy_rate_pct",
                    "multidimensional_poverty_index",
                ]
            ]
            efficiency_features = scenario_input[
                [
                    "month_num",
                    "is_festive_season",
                    "city",
                    "program",
                    "donor_type",
                    "channel",
                    "expense_amount",
                    "volunteer_hours",
                    "event_count",
                    "social_reach",
                    "literacy_rate_pct",
                    "multidimensional_poverty_index",
                ]
            ]

            predicted_donation = float(donation_model.predict(donation_features)[0])
            predicted_beneficiaries = float(beneficiary_model.predict(beneficiary_features)[0])
            efficiency_probability = None
            if efficiency_model is not None and hasattr(efficiency_model, "predict_proba"):
                efficiency_probability = float(efficiency_model.predict_proba(efficiency_features)[0][1])

            result_cols = st.columns(3)
            with result_cols[0]:
                metric_card("Predicted Donations", money(predicted_donation), "Next-month estimate")
            with result_cols[1]:
                metric_card("Predicted Beneficiaries", compact_number(predicted_beneficiaries), "Expected reach")
            with result_cols[2]:
                if efficiency_probability is not None:
                    metric_card(
                        "High-Efficiency Probability",
                        f"{efficiency_probability * 100:.1f}%",
                        "Likely cost-efficient delivery",
                    )

            predicted_cost = scenario_expense / max(predicted_beneficiaries, 1)
            st.markdown(
                f'<div class="insight-box">With {scenario_hours:.0f} volunteer hours and {money(scenario_expense)} '
                f"planned spend in {scenario_city}, the model expects roughly "
                f"{compact_number(predicted_beneficiaries)} beneficiaries and {money(predicted_donation)} in donations, "
                f"at an estimated {money(predicted_cost)} per beneficiary.</div>",
                unsafe_allow_html=True,
            )

with context_tab:
    context_df = load_context_data()
    if context_df.empty:
        st.info("State context dataset not found.")
    else:
        city_context = (
            filtered.groupby(["city", "state"], as_index=False)
            .agg(
                beneficiaries_reached=("beneficiaries_reached", "sum"),
                volunteer_hours=("volunteer_hours", "sum"),
                donation_amount=("donation_amount", "sum"),
            )
        )
        merged_context = city_context.merge(context_df, on="state", how="left")
        merged_context["volunteers_per_1000_beneficiaries"] = (
            merged_context["volunteer_hours"] / merged_context["beneficiaries_reached"] * 1000
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                px.bar(
                    merged_context.sort_values("literacy_rate_pct"),
                    x="literacy_rate_pct",
                    y="city",
                    color="beneficiaries_reached",
                    orientation="h",
                    title="Literacy Context vs NayePankh Reach by City",
                    color_continuous_scale=["#D1FAE5", "#0F9F8F"],
                ).update_layout(margin=dict(l=20, r=20, t=55, b=20), xaxis_title="Literacy Rate (%)", yaxis_title=""),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                px.scatter(
                    merged_context,
                    x="multidimensional_poverty_index",
                    y="volunteers_per_1000_beneficiaries",
                    size="beneficiaries_reached",
                    color="city",
                    title="Need vs Volunteer Support",
                    color_discrete_sequence=ACCENT_COLORS,
                    hover_data={"donation_amount": ":,.0f"},
                ).update_layout(
                    xaxis_title="Multidimensional Poverty Index",
                    yaxis_title="Volunteer Hours per 1,000 Beneficiaries",
                    margin=dict(l=20, r=20, t=55, b=20),
                ),
                use_container_width=True,
            )

        st.dataframe(
            merged_context.assign(
                donation_amount=merged_context["donation_amount"].map(money),
                beneficiaries_reached=merged_context["beneficiaries_reached"].map(compact_number),
                volunteers_per_1000_beneficiaries=merged_context["volunteers_per_1000_beneficiaries"].map(
                    lambda x: f"{x:.1f}"
                ),
            ),
            hide_index=True,
            use_container_width=True,
        )

        high_need = merged_context.sort_values(
            ["multidimensional_poverty_index", "volunteers_per_1000_beneficiaries"],
            ascending=[False, True],
        ).iloc[0]
        st.markdown(
            f'<div class="insight-box">{high_need["city"]} combines higher regional need with comparatively lower volunteer support, making it a strong candidate for additional field allocation.</div>',
            unsafe_allow_html=True,
        )

with report_tab:
    report_text = build_report(filtered)
    st.markdown(report_text)
    st.download_button(
        "Download Summary Report",
        data=report_text,
        file_name="naye_pankh_impact_summary.md",
        mime="text/markdown",
    )
    st.download_button(
        "Download Filtered Dataset",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_naye_pankh_impact_data.csv",
        mime="text/csv",
    )
