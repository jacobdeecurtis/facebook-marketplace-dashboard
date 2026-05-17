import base64
import mimetypes
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Auction Dashboard", layout="wide")

APP_DIR = Path(__file__).parent
HEADER_IMAGE_WIDTH = "min(50vw, 700px)"
HEADER_IMAGE_PATHS = (
    APP_DIR / "boxland.png",
    APP_DIR / "assets" / "boxland.png",
    APP_DIR / "assets" / "boxland2.png",
)

SHEET_ID = "1UkYDjeRaJlu3ByJYLeKzBScu7OwY6vxd2BgU-EnT41I"
SALES_GID = "900908138"
LISTINGS_GID = "944629416"
AUCTIONS_GID = "1075555494"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={SALES_GID}"
LISTINGS_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={LISTINGS_GID}"
AUCTIONS_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={AUCTIONS_GID}"
AUCTION_TAB_EXCLUDED_DATES = pd.to_datetime(["2026-05-14"])


def money_to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


@st.cache_data(ttl=3600)
def import_auction_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads Marketplace sales, auction costs, and other reselling costs from Google Sheets."""
    df_google_sheet = pd.read_csv(CSV_URL)
    df_costs = pd.read_csv(AUCTIONS_CSV_URL)

    required_auction_columns = {"Auction_Date", "Spent"}
    missing_auction_columns = required_auction_columns - set(df_costs.columns)
    if missing_auction_columns:
        missing = ", ".join(sorted(missing_auction_columns))
        raise ValueError(f"Auctions sheet is missing required column(s): {missing}")

    df_google_sheet["Auction_Date"] = pd.to_datetime(df_google_sheet["Auction_Date"], format="%m/%d/%Y", errors="coerce")
    df_google_sheet["sale_date"] = pd.to_datetime(df_google_sheet["sale_date"], format="%m/%d/%Y", errors="coerce")
    df_google_sheet["revenue"] = money_to_number(df_google_sheet["revenue"])

    df_costs["Auction_Date"] = pd.to_datetime(df_costs["Auction_Date"], format="%m/%d/%Y", errors="coerce")
    df_costs["auction_cost"] = money_to_number(df_costs["Spent"])
    df_costs = (
        df_costs.dropna(subset=["Auction_Date"])[["Auction_Number", "Auction_Date", "auction_cost"]]
        .copy()
    )
    df_costs["Auction_Number"] = df_costs["Auction_Number"].astype(str).str.strip()
    df_auction_cost = df_costs[df_costs["Auction_Number"].ne("") & df_costs["Auction_Number"].ne("nan")].copy()
    df_other_cost = df_costs[df_costs["Auction_Number"].eq("") | df_costs["Auction_Number"].eq("nan")].copy()

    merged_df = pd.merge(df_google_sheet, df_auction_cost, on="Auction_Date", how="left")
    return merged_df, df_auction_cost, df_other_cost


@st.cache_data(ttl=3600)
def import_listings_data() -> pd.DataFrame:
    """Loads Marketplace listing data from the Listings Google Sheet tab."""
    df_listings = pd.read_csv(LISTINGS_CSV_URL)
    if "Listing_Date" not in df_listings.columns:
        return pd.DataFrame(columns=["Listing_Date"])

    if "Auction_Date" in df_listings.columns:
        df_listings["Auction_Date"] = pd.to_datetime(df_listings["Auction_Date"], errors="coerce")
    df_listings["Listing_Date"] = pd.to_datetime(df_listings["Listing_Date"], errors="coerce")
    df_listings = df_listings.dropna(subset=["Listing_Date"]).copy()
    df_listings["Listing_Date"] = df_listings["Listing_Date"].dt.normalize()
    return df_listings


def daily_listing_counts(df_listings: pd.DataFrame, plot_start_date: pd.Timestamp) -> pd.DataFrame:
    """Returns one row per calendar day with the number of listings posted that day."""
    if df_listings.empty or "Listing_Date" not in df_listings.columns:
        return pd.DataFrame(columns=["Listing_Date", "number_of_listings"])

    listing_dates = df_listings.dropna(subset=["Listing_Date"]).copy()
    if listing_dates.empty:
        return pd.DataFrame(columns=["Listing_Date", "number_of_listings"])

    daily_counts = (
        listing_dates.groupby("Listing_Date")
        .size()
        .reset_index(name="number_of_listings")
    )
    full_date_range = pd.date_range(
        start=listing_dates["Listing_Date"].min(),
        end=listing_dates["Listing_Date"].max(),
        freq="D",
    )
    all_daily_listings = (
        pd.DataFrame({"Listing_Date": full_date_range})
        .merge(daily_counts, on="Listing_Date", how="left")
    )
    all_daily_listings["number_of_listings"] = (
        all_daily_listings["number_of_listings"].fillna(0).astype(int)
    )
    return all_daily_listings[all_daily_listings["Listing_Date"] >= plot_start_date]


def build_daily_listings_figure(daily_listings: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=daily_listings["Listing_Date"],
                y=daily_listings["number_of_listings"],
                mode="lines+markers+text",
                name="Number of Listings",
                text=daily_listings["number_of_listings"],
                textposition="top center",
                hovertemplate=(
                    "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                    "<b>Listings</b>: %{y}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title_text="Number of Listings Per Day Starting 2026-04-30",
        xaxis_title="Date",
        yaxis_title="Number of Listings",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(tickformat="%Y-%m-%d", dtick="D1"),
    )
    return fig


def auction_to_listing_records(df_listings: pd.DataFrame) -> pd.DataFrame:
    """Returns listings with a valid, non-negative age from auction date to listing date."""
    required_columns = {"Auction_Date", "Listing_Date"}
    if df_listings.empty or not required_columns.issubset(df_listings.columns):
        return pd.DataFrame(columns=["Auction_Date", "Listing_Date", "days_from_auction_to_listing"])

    records = df_listings.dropna(subset=["Auction_Date", "Listing_Date"]).copy()
    if records.empty:
        return pd.DataFrame(columns=["Auction_Date", "Listing_Date", "days_from_auction_to_listing"])

    records["Auction_Date"] = records["Auction_Date"].dt.normalize()
    records["Listing_Date"] = records["Listing_Date"].dt.normalize()
    records["days_from_auction_to_listing"] = (
        records["Listing_Date"] - records["Auction_Date"]
    ).dt.days
    return records[records["days_from_auction_to_listing"] >= 0].copy()


def build_auction_to_listing_distribution(records: pd.DataFrame) -> go.Figure:
    max_days = int(records["days_from_auction_to_listing"].max())
    timing_days = records["days_from_auction_to_listing"]
    mean_days = timing_days.mean()
    median_days = timing_days.median()
    std_days = timing_days.std(ddof=0)
    fig = px.histogram(
        records,
        x="days_from_auction_to_listing",
        nbins=max(5, min(20, max_days + 1)),
        title="Distribution of Days from Auction to Listing",
        hover_data=[column for column in ["Title", "Auction_Date", "Listing_Date"] if column in records.columns],
    )
    fig.update_layout(
        xaxis_title="Days from auction to listing",
        yaxis_title="Listings",
        bargap=0.05,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    line_specs = [
        ("Mean", mean_days, "#EF553B", "dash"),
        ("Median", median_days, "#00CC96", "dot"),
    ]
    if pd.notna(std_days):
        line_specs.extend([
            ("Mean - 1 SD", max(0, mean_days - std_days), "#AB63FA", "dashdot"),
            ("Mean + 1 SD", mean_days + std_days, "#AB63FA", "dashdot"),
        ])
    for label, value, color, dash in line_specs:
        fig.add_vline(
            x=value,
            line_width=2,
            line_dash=dash,
            line_color=color,
            annotation_text=f"{label}: {value:.1f}",
            annotation_position="top right",
        )
    return fig


def daily_auction_to_listing_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Returns auction-to-listing timing by listing date."""
    if records.empty:
        return pd.DataFrame(columns=[
            "Listing_Date",
            "listings",
            "avg_days_from_auction_to_listing",
            "median_days_from_auction_to_listing",
            "min_days_from_auction_to_listing",
            "max_days_from_auction_to_listing",
        ])

    summary = (
        records.groupby("Listing_Date", as_index=False)
        .agg(
            listings=("days_from_auction_to_listing", "count"),
            avg_days_from_auction_to_listing=("days_from_auction_to_listing", "mean"),
            median_days_from_auction_to_listing=("days_from_auction_to_listing", "median"),
            min_days_from_auction_to_listing=("days_from_auction_to_listing", "min"),
            max_days_from_auction_to_listing=("days_from_auction_to_listing", "max"),
        )
        .sort_values("Listing_Date", ascending=False)
    )
    return summary


def normalize_match_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )


def listing_to_sale_records(df_sales: pd.DataFrame, df_listings: pd.DataFrame) -> pd.DataFrame:
    """Matches sold items to listing rows and returns days from listing to sale."""
    required_sales = {"Auction_Date", "Title", "sale_date"}
    required_listings = {"Auction_Date", "Title", "Listing_Date"}
    if (
        df_sales.empty
        or df_listings.empty
        or not required_sales.issubset(df_sales.columns)
        or not required_listings.issubset(df_listings.columns)
    ):
        return pd.DataFrame(columns=[
            "sale_date",
            "Listing_Date",
            "Title",
            "product_category",
            "revenue",
            "days_to_sale",
        ])

    sales = df_sales.dropna(subset=["Auction_Date", "Title", "sale_date"]).copy()
    listings = df_listings.dropna(subset=["Auction_Date", "Title", "Listing_Date"]).copy()
    if sales.empty or listings.empty:
        return pd.DataFrame(columns=[
            "sale_date",
            "Listing_Date",
            "Title",
            "product_category",
            "revenue",
            "days_to_sale",
        ])

    sales["_match_title"] = normalize_match_text(sales["Title"])
    listings["_match_title"] = normalize_match_text(listings["Title"])
    sales["_match_auction_date"] = sales["Auction_Date"].dt.normalize()
    listings["_match_auction_date"] = listings["Auction_Date"].dt.normalize()
    sales = sales.sort_values(["_match_auction_date", "_match_title", "sale_date"])
    listings = listings.sort_values(["_match_auction_date", "_match_title", "Listing_Date"])
    match_keys = ["_match_auction_date", "_match_title"]
    sales["_match_number"] = sales.groupby(match_keys).cumcount()
    listings["_match_number"] = listings.groupby(match_keys).cumcount()

    matched = sales.merge(
        listings[match_keys + ["_match_number", "Listing_Date"]],
        on=match_keys + ["_match_number"],
        how="inner",
    )
    matched["days_to_sale"] = (matched["sale_date"] - matched["Listing_Date"]).dt.days
    matched = matched[matched["days_to_sale"] >= 0].copy()
    return matched.drop(columns=["_match_title", "_match_auction_date", "_match_number"], errors="ignore")


def weekly_listing_to_sale(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=[
            "week_start",
            "week_end",
            "week_label",
            "avg_days_to_sale",
            "items_sold",
            "avg_days_label",
        ])

    weekly = records.copy()
    weekly["week_start"] = weekly["sale_date"].dt.to_period("W-SUN").apply(lambda period: period.start_time)
    weekly = (
        weekly.groupby("week_start", as_index=False)
        .agg(
            avg_days_to_sale=("days_to_sale", "mean"),
            items_sold=("days_to_sale", "count"),
        )
        .sort_values("week_start")
    )
    weekly["week_end"] = weekly["week_start"] + pd.Timedelta(days=6)
    weekly["week_label"] = weekly.apply(format_week_label, axis=1)
    weekly["avg_days_label"] = weekly["avg_days_to_sale"].map(lambda value: f"{value:.1f}")
    return weekly


def build_avg_listing_to_sale_figure(weekly_days_to_sale: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        weekly_days_to_sale,
        x="week_label",
        y="avg_days_to_sale",
        text="avg_days_label",
        title="Average Days from Listing to Sale by Sale Week",
        hover_data={
            "week_label": True,
            "avg_days_to_sale": ":.1f",
            "items_sold": True,
        },
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        xaxis_title="Sale week",
        yaxis_title="Average days",
        xaxis_tickangle=-45,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_listing_to_sale_distribution(records: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        records,
        x="days_to_sale",
        nbins=max(5, min(20, int(records["days_to_sale"].max()) + 1)),
        title="Distribution of Days from Listing to Sale",
        hover_data=["Title", "sale_date", "Listing_Date"],
    )
    fig.update_layout(
        xaxis_title="Days from listing to sale",
        yaxis_title="Items sold",
        bargap=0.05,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def summarize_by_auction(df: pd.DataFrame, df_auction_cost: pd.DataFrame) -> pd.DataFrame:
    revenue_per_auction = (
        df.dropna(subset=["Auction_Date"])
        .groupby("Auction_Date", as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "total_revenue"})
    )
    summary = pd.merge(revenue_per_auction, df_auction_cost, on="Auction_Date", how="outer")
    summary["total_revenue"] = summary["total_revenue"].fillna(0)
    summary["auction_cost"] = summary["auction_cost"].fillna(0)
    summary["profit"] = summary["total_revenue"] - summary["auction_cost"]
    summary["profit_to_cost"] = np.where(summary["auction_cost"] > 0, summary["profit"] / summary["auction_cost"], np.nan)
    summary = summary.sort_values("Auction_Date")
    cumulative_cost = summary["auction_cost"].cumsum()
    cumulative_profit = summary["profit"].cumsum()
    summary["cumulative_profit_to_cost"] = np.where(cumulative_cost > 0, cumulative_profit / cumulative_cost, np.nan)
    return summary


def filter_auction_tab_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Hides non-auction cost dates from the auction performance tab only."""
    if summary.empty:
        return summary

    excluded_dates = AUCTION_TAB_EXCLUDED_DATES.normalize()
    auction_dates = summary["Auction_Date"].dt.normalize()
    return summary[~auction_dates.isin(excluded_dates)].copy()


def format_week_label(row: pd.Series) -> str:
    start = row["week_start"]
    end = row["week_end"]
    start_label = f"{start:%b} {start.day}"
    end_label = f"{end:%b} {end.day}"
    return f"{start_label}-{end_label}"


def weekly_sales(df: pd.DataFrame) -> pd.DataFrame:
    weekly = df.dropna(subset=["sale_date"]).copy()
    if weekly.empty:
        return pd.DataFrame(columns=["week_start", "week_end", "week_label", "revenue", "items_sold", "revenue_label"])

    weekly["week_start"] = weekly["sale_date"].dt.to_period("W-SUN").apply(lambda period: period.start_time)
    weekly = (
        weekly.groupby("week_start", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            items_sold=("revenue", "count"),
        )
        .sort_values("week_start")
    )
    weekly["week_end"] = weekly["week_start"] + pd.Timedelta(days=6)
    weekly["week_label"] = weekly.apply(format_week_label, axis=1)
    weekly["revenue_label"] = weekly["revenue"].map(lambda value: f"${value:,.0f}")
    return weekly


def daily_profit(df: pd.DataFrame, df_costs: pd.DataFrame) -> pd.DataFrame:
    sales = (
        df.dropna(subset=["sale_date"])
        .groupby("sale_date", as_index=False)["revenue"]
        .sum()
        .rename(columns={"sale_date": "date", "revenue": "daily_revenue"})
    )
    costs = (
        df_costs.dropna(subset=["Auction_Date"])
        .groupby("Auction_Date", as_index=False)["auction_cost"]
        .sum()
        .rename(columns={"Auction_Date": "date", "auction_cost": "daily_auction_cost"})
    )

    if sales.empty and costs.empty:
        return pd.DataFrame(columns=[
            "date",
            "daily_revenue",
            "daily_auction_cost",
            "daily_net_profit",
            "cumulative_net_profit",
        ])

    min_date = min(
        value
        for value in [
            sales["date"].min() if not sales.empty else pd.NaT,
            costs["date"].min() if not costs.empty else pd.NaT,
        ]
        if pd.notna(value)
    )
    max_date = max(
        value
        for value in [
            sales["date"].max() if not sales.empty else pd.NaT,
            costs["date"].max() if not costs.empty else pd.NaT,
        ]
        if pd.notna(value)
    )

    daily = pd.DataFrame({"date": pd.date_range(start=min_date, end=max_date, freq="D")})
    daily = daily.merge(sales, on="date", how="left").merge(costs, on="date", how="left")
    daily[["daily_revenue", "daily_auction_cost"]] = daily[["daily_revenue", "daily_auction_cost"]].fillna(0)
    daily["daily_net_profit"] = daily["daily_revenue"] - daily["daily_auction_cost"]
    daily["cumulative_net_profit"] = daily["daily_net_profit"].cumsum()
    return daily


def build_cumulative_profit_projection(daily: pd.DataFrame) -> go.Figure:
    projection_target_date = datetime(2026, 8, 19)
    actual = daily.sort_values("date").copy()
    projection_end_date = max(projection_target_date, actual["date"].max())

    fig = px.line(
        actual,
        x="date",
        y="cumulative_net_profit",
        title="Overall Cumulative Net Profit Over Time with Linear and Polynomial Regression Projections",
    )
    fig.update_traces(name="Actual", showlegend=True, line=dict(color="#636EFA"))

    if len(actual) >= 2:
        actual["date_ordinal"] = actual["date"].map(datetime.toordinal)
        x = actual["date_ordinal"].to_numpy(dtype=float)
        y = actual["cumulative_net_profit"].to_numpy(dtype=float)
        x_anchor = x.min()
        x_centered = x - x_anchor
        projection_dates = pd.date_range(start=actual["date"].min(), end=projection_end_date, freq="D")
        projection_x = np.array([date.toordinal() for date in projection_dates], dtype=float) - x_anchor

        linear_coefficients = np.polyfit(x_centered, y, deg=1)
        predicted_profit_linear = np.polyval(linear_coefficients, projection_x)

        polynomial_degree = min(2, len(actual) - 1)
        polynomial_coefficients = np.polyfit(x_centered, y, deg=polynomial_degree)
        predicted_profit_polynomial = np.polyval(polynomial_coefficients, projection_x)

        fig.add_trace(go.Scatter(
            x=projection_dates,
            y=predicted_profit_linear,
            mode="lines",
            name="Linear",
            line=dict(dash="dot", color="red"),
        ))
        fig.add_trace(go.Scatter(
            x=projection_dates,
            y=predicted_profit_polynomial,
            mode="lines",
            name="Polynomial",
            line=dict(dash="dot", color="blue"),
        ))

        peaks = actual[
            (actual["cumulative_net_profit"].shift(1) < actual["cumulative_net_profit"])
            & (actual["cumulative_net_profit"].shift(-1) < actual["cumulative_net_profit"])
        ].copy()
        if peaks.empty or len(peaks) < 2:
            last_point = actual.iloc[[-1]].copy()
            if peaks.empty or not (peaks["date"] == last_point["date"].iloc[0]).any():
                peaks = pd.concat([peaks, last_point])

        if not peaks.empty:
            fig.add_trace(go.Scatter(
                x=peaks["date"],
                y=peaks["cumulative_net_profit"],
                mode="text",
                text=peaks["cumulative_net_profit"].map(lambda value: f"${value:,.0f}"),
                textposition="top center",
                showlegend=False,
                name="Peaks",
                textfont=dict(color="darkred", size=10),
            ))

        estimated_profit_linear = predicted_profit_linear[-1]
        estimated_profit_polynomial = predicted_profit_polynomial[-1]
        fig.add_trace(go.Scatter(
            x=[projection_end_date],
            y=[estimated_profit_linear],
            mode="text",
            text=[f"Linear: ${estimated_profit_linear:,.0f}"],
            textposition="bottom right",
            showlegend=False,
            name="Linear Estimate",
            textfont=dict(color="red", size=10, weight="bold"),
        ))
        fig.add_trace(go.Scatter(
            x=[projection_end_date],
            y=[estimated_profit_polynomial],
            mode="text",
            text=[f"Poly: ${estimated_profit_polynomial:,.0f}"],
            textposition="top right",
            showlegend=False,
            name="Polynomial Estimate",
            textfont=dict(color="blue", size=10, weight="bold"),
        ))

    last_actual = actual.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_actual["date"]],
        y=[last_actual["cumulative_net_profit"]],
        mode="text",
        text=[f"Actual: ${last_actual['cumulative_net_profit']:,.0f}"],
        textposition="bottom center",
        showlegend=False,
        name="Last Actual Profit",
        textfont=dict(color="green", size=10, weight="bold"),
    ))

    fig.add_shape(
        type="line",
        x0=actual["date"].min(),
        y0=0,
        x1=projection_end_date,
        y1=0,
        line=dict(color="Grey", width=2, dash="dash"),
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Net Profit",
        hovermode="x unified",
        yaxis=dict(tickformat="$,.0f"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig

def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "product_category" not in data.columns:
        data["product_category"] = "Uncategorized"
    data["days_to_sell"] = (data["sale_date"] - data["Auction_Date"]).dt.days
    return (
        data.dropna(subset=["product_category"])
        .groupby("product_category", as_index=False)
        .agg(
            items_sold=("revenue", "count"),
            total_revenue=("revenue", "sum"),
            avg_selling_price=("revenue", "mean"),
            avg_days_to_sell=("days_to_sell", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
    )

def find_header_image() -> Path | None:
    return next((path for path in HEADER_IMAGE_PATHS if path.exists()), None)


def image_data_uri(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


def render_header() -> None:
    header_image = find_header_image()
    if header_image is None:
        st.warning("Header image `boxland.png` was not found next to `app.py` or in `assets/`.")
        return

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
            <img src="{image_data_uri(header_image)}"
                 alt="Boxland logo"
                 style="width: {HEADER_IMAGE_WIDTH}; max-width: 100vw; height: auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


render_header()
st.caption("Auto-refreshes from Google Sheets. Data is cached for 1 hour.")

with st.sidebar:
    st.header("Filters")
    refresh = st.button("Refresh now")
    if refresh:
        st.cache_data.clear()

try:
    df, df_auction_cost, df_other_cost = import_auction_data()
except Exception as e:
    st.error("Could not load the Google Sheet. Make sure the sheet is shared/public or accessible as CSV.")
    st.exception(e)
    st.stop()

min_date = df["sale_date"].min()
max_date = df["sale_date"].max()

with st.sidebar:
    date_range = st.date_input(
        "Sale date range",
        value=(min_date.date() if pd.notna(min_date) else datetime.today().date(), max_date.date() if pd.notna(max_date) else datetime.today().date()),
    )

filtered = df.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[(filtered["sale_date"].isna()) | ((filtered["sale_date"] >= start) & (filtered["sale_date"] <= end))]

auction_summary = summarize_by_auction(filtered, df_auction_cost)
auction_tab_summary = filter_auction_tab_summary(auction_summary)
weekly = weekly_sales(filtered)
df_total_cost = pd.concat([df_auction_cost, df_other_cost], ignore_index=True)

listings_load_error = None
try:
    df_listings = import_listings_data()
except Exception as e:
    listings_load_error = e
    df_listings = pd.DataFrame()

listing_sale_records = listing_to_sale_records(filtered, df_listings)
avg_listing_to_sale_days = (
    listing_sale_records["days_to_sale"].mean() if not listing_sale_records.empty else np.nan
)
auction_listing_records = auction_to_listing_records(df_listings)

sold_items = filtered[filtered["sale_date"].notna()].copy()
total_revenue = sold_items["revenue"].sum()
total_cost = df_total_cost["auction_cost"].sum()
total_profit = total_revenue - total_cost
items_sold = sold_items["revenue"].count()
avg_sale_price = sold_items["revenue"].mean() if items_sold else 0
profit_to_cost = total_profit / total_cost if total_cost else np.nan

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Revenue", f"${total_revenue:,.0f}")
c2.metric("Total Cost", f"${total_cost:,.0f}")
c3.metric("Profit", f"${total_profit:,.0f}")
c4.metric("Items Sold", f"{items_sold:,.0f}")
c5.metric("Profit / Cost", "N/A" if np.isnan(profit_to_cost) else f"{profit_to_cost:.1%}")
c6.metric(
    "Avg List to Sale",
    "N/A" if np.isnan(avg_listing_to_sale_days) else f"{avg_listing_to_sale_days:.1f} days",
)

st.divider()

tab_overview, tab_auctions, tab_listings, tab_categories, tab_recent, tab_raw = st.tabs([
    "Overview", "By auction", "Listings", "Categories", "Recent sales", "Raw data"
])

with tab_overview:
    st.subheader("Sales over time")
    if not weekly.empty:
        fig_weekly = px.bar(
            weekly,
            x="week_label",
            y="revenue",
            text="revenue_label",
            title="Weekly Revenue",
            hover_data={"week_label": True, "revenue": ":$,.0f", "items_sold": True},
        )
        fig_weekly.update_traces(textposition="outside", cliponaxis=False)
        fig_weekly.update_layout(yaxis_title="Revenue", xaxis_title="Week", xaxis_tickangle=-45)
        st.plotly_chart(fig_weekly, use_container_width=True)

    else:
        st.info("No sales found for this date range.")

    st.subheader("Daily cumulative profit")
    profit_daily = daily_profit(filtered, df_total_cost)
    if not profit_daily.empty:
        st.plotly_chart(build_cumulative_profit_projection(profit_daily), use_container_width=True)
    else:
        st.info("No profit data found for this date range.")

with tab_auctions:
    st.subheader("Auction performance")
    fig = go.Figure()
    fig.add_bar(x=auction_tab_summary["Auction_Date"], y=auction_tab_summary["total_revenue"], name="Revenue")
    fig.add_bar(x=auction_tab_summary["Auction_Date"], y=auction_tab_summary["auction_cost"], name="Cost")
    fig.add_scatter(x=auction_tab_summary["Auction_Date"], y=auction_tab_summary["profit"], name="Profit", mode="lines+markers")
    fig.update_layout(barmode="group", xaxis_title="Auction Date", yaxis_title="Dollars")
    st.plotly_chart(fig, use_container_width=True)

    display_summary = auction_tab_summary.copy()
    st.dataframe(
        display_summary.style.format({
            "total_revenue": "${:,.0f}",
            "auction_cost": "${:,.0f}",
            "profit": "${:,.0f}",
            "profit_to_cost": "{:.1%}",
            "cumulative_profit_to_cost": "{:.1%}",
        }),
        use_container_width=True,
    )

with tab_listings:
    st.subheader("Listings per day")
    plot_start_date = pd.to_datetime("2026-04-30")

    if listings_load_error is not None:
        st.error("Could not load the Listings Google Sheet tab. Make sure the sheet is shared/public or accessible as CSV.")
        st.exception(listings_load_error)

    elif not df_listings.empty:
        daily_listings = daily_listing_counts(df_listings, plot_start_date)
        if not daily_listings.empty:
            st.plotly_chart(build_daily_listings_figure(daily_listings), use_container_width=True)
            st.dataframe(daily_listings, use_container_width=True)
        else:
            st.info(f"No listings found on or after {plot_start_date:%Y-%m-%d}.")
    else:
        st.info("Listings data is empty, so there is nothing to chart.")

    st.subheader("Auction to listing timing")
    if not auction_listing_records.empty:
        auction_listing_days = auction_listing_records["days_from_auction_to_listing"]
        auction_listing_metric_cols = st.columns(4)
        auction_listing_metric_cols[0].metric("Matched Listings", f"{len(auction_listing_records):,.0f}")
        auction_listing_metric_cols[1].metric("Mean Days", f"{auction_listing_days.mean():.1f}")
        auction_listing_metric_cols[2].metric("Median Days", f"{auction_listing_days.median():.1f}")
        auction_listing_metric_cols[3].metric("Max Days", f"{auction_listing_days.max():.0f}")
        st.plotly_chart(build_auction_to_listing_distribution(auction_listing_records), use_container_width=True)

        st.write("Daily auction-to-listing timing")
        daily_auction_listing = daily_auction_to_listing_summary(auction_listing_records)
        st.dataframe(
            daily_auction_listing.style.format({
                "avg_days_from_auction_to_listing": "{:.1f}",
                "median_days_from_auction_to_listing": "{:.1f}",
                "min_days_from_auction_to_listing": "{:.0f}",
                "max_days_from_auction_to_listing": "{:.0f}",
            }),
            use_container_width=True,
        )

        st.write("Listing detail")
        display_auction_listing_records = auction_listing_records[
            [
                column
                for column in [
                    "Listing_Date",
                    "Auction_Date",
                    "Title",
                    "product_category",
                    "days_from_auction_to_listing",
                ]
                if column in auction_listing_records.columns
            ]
        ].sort_values(["Listing_Date", "days_from_auction_to_listing"], ascending=[False, False])
        st.dataframe(display_auction_listing_records, use_container_width=True)
    else:
        st.info("No listings with both an auction date and listing date were found.")

    st.subheader("Listing to sale timing")
    if not listing_sale_records.empty:
        weekly_days_to_sale = weekly_listing_to_sale(listing_sale_records)
        timing_metric_cols = st.columns(3)
        timing_metric_cols[0].metric("Matched Sold Items", f"{len(listing_sale_records):,.0f}")
        timing_metric_cols[1].metric("Average Days", f"{avg_listing_to_sale_days:.1f}")
        timing_metric_cols[2].metric("Median Days", f"{listing_sale_records['days_to_sale'].median():.1f}")
        if not weekly_days_to_sale.empty:
            st.plotly_chart(build_avg_listing_to_sale_figure(weekly_days_to_sale), use_container_width=True)
        st.plotly_chart(build_listing_to_sale_distribution(listing_sale_records), use_container_width=True)
        display_listing_sale_records = listing_sale_records[
            [
                column
                for column in [
                    "sale_date",
                    "Listing_Date",
                    "Title",
                    "product_category",
                    "revenue",
                    "days_to_sale",
                ]
                if column in listing_sale_records.columns
            ]
        ].sort_values("sale_date", ascending=False)
        st.dataframe(display_listing_sale_records, use_container_width=True)
    else:
        st.info("No sold listings with both a listing date and sale date were found for this sale date range.")

with tab_categories:
    st.subheader("Category performance")
    cats = category_summary(filtered)
    if not cats.empty:
        fig = px.bar(cats.head(15), x="product_category", y="total_revenue", title="Top categories by revenue")
        fig.update_layout(xaxis_title="Category", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            cats.style.format({
                "total_revenue": "${:,.0f}",
                "avg_selling_price": "${:,.0f}",
                "avg_days_to_sell": "{:.1f}",
            }),
            use_container_width=True,
        )
    else:
        st.info("No category data found.")

with tab_recent:
    st.subheader("Recent sales")
    cols = [c for c in ["sale_date", "Title", "product", "product_category", "revenue", "Auction_Date"] if c in filtered.columns]
    recent = filtered[filtered["sale_date"].notna()].sort_values("sale_date", ascending=False)[cols].head(50)
    st.dataframe(recent, use_container_width=True)

with tab_raw:
    st.subheader("Raw merged data")
    st.dataframe(filtered, use_container_width=True)
