import base64
from html import escape
import mimetypes
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
DONATIONS_GID = "152860955"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={SALES_GID}"
LISTINGS_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={LISTINGS_GID}"
AUCTIONS_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={AUCTIONS_GID}"
DONATIONS_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={DONATIONS_GID}"
AUCTION_TAB_EXCLUDED_DATES = pd.to_datetime(["2026-05-14"])
CUMULATIVE_PROFIT_EXCLUDED_DATES = pd.to_datetime(["2025-01-01", "2026-01-01"])
TAX_YEAR = 2026


def money_to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def format_date_for_display(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.to_datetime(value).strftime("%Y-%m-%d")


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
    auction_number = df_costs["Auction_Number"].astype("string").str.strip()
    has_auction_number = auction_number.notna() & auction_number.ne("") & auction_number.str.casefold().ne("nan")
    df_costs["Auction_Number"] = auction_number.where(has_auction_number, "").astype(str)
    df_auction_cost = df_costs[has_auction_number].copy()
    df_other_cost = df_costs[~has_auction_number].copy()

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


@st.cache_data(ttl=3600)
def import_donations_data() -> pd.DataFrame:
    """Loads tax-deductible donation records from the Donations Google Sheet tab."""
    df_donations = pd.read_csv(DONATIONS_CSV_URL)
    required_columns = {"Donated", "Value"}
    missing_columns = required_columns - set(df_donations.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Donations sheet is missing required column(s): {missing}")

    df_donations["Donated"] = pd.to_datetime(df_donations["Donated"], errors="coerce")
    df_donations["Value"] = money_to_number(df_donations["Value"])
    if "Received date" in df_donations.columns:
        df_donations["Received date"] = pd.to_datetime(df_donations["Received date"], errors="coerce")
    return df_donations


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


def prepare_cumulative_profit_data(df_sales: pd.DataFrame, df_auction_cost: pd.DataFrame) -> pd.DataFrame:
    """Returns day-by-day cumulative profit for each auction through the latest sale date."""
    required_sales_columns = {"Auction_Date", "sale_date", "revenue"}
    required_cost_columns = {"Auction_Date", "auction_cost"}
    if (
        df_sales.empty
        or df_auction_cost.empty
        or not required_sales_columns.issubset(df_sales.columns)
        or not required_cost_columns.issubset(df_auction_cost.columns)
    ):
        return pd.DataFrame(columns=[
            "Auction_Date",
            "sale_date",
            "daily_revenue_sum",
            "cumulative_revenue",
            "cumulative_profit",
            "time_since_auction",
            "auction_cost",
        ])

    sales = df_sales.copy()
    sales["Auction_Date"] = pd.to_datetime(sales["Auction_Date"], errors="coerce").dt.normalize()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce").dt.normalize()
    sales["revenue"] = pd.to_numeric(sales["revenue"], errors="coerce")
    sales = sales.dropna(subset=["Auction_Date", "sale_date", "revenue"]).copy()
    if sales.empty:
        return pd.DataFrame(columns=[
            "Auction_Date",
            "sale_date",
            "daily_revenue_sum",
            "cumulative_revenue",
            "cumulative_profit",
            "time_since_auction",
            "auction_cost",
        ])

    auction_costs = df_auction_cost.copy()
    auction_costs["Auction_Date"] = pd.to_datetime(auction_costs["Auction_Date"], errors="coerce").dt.normalize()
    auction_costs["auction_cost"] = pd.to_numeric(auction_costs["auction_cost"], errors="coerce").fillna(0)
    auction_costs = (
        auction_costs.dropna(subset=["Auction_Date"])
        .groupby("Auction_Date", as_index=False)["auction_cost"]
        .sum()
    )

    latest_sale_date = sales["sale_date"].max()
    cumulative_rows = []
    for auction_date in sorted(sales["Auction_Date"].dropna().unique()):
        auction_sales = sales[sales["Auction_Date"] == auction_date].copy()
        daily_revenue = auction_sales.groupby("sale_date", as_index=False)["revenue"].sum()
        if latest_sale_date < auction_date:
            continue

        full_dates = pd.DataFrame({"sale_date": pd.date_range(start=auction_date, end=latest_sale_date, freq="D")})
        full_dates["Auction_Date"] = auction_date
        auction_cost_row = auction_costs[auction_costs["Auction_Date"] == auction_date]
        auction_cost = 0 if auction_cost_row.empty else auction_cost_row["auction_cost"].iloc[0]

        auction_daily = full_dates.merge(daily_revenue, on="sale_date", how="left")
        auction_daily["daily_revenue_sum"] = auction_daily["revenue"].fillna(0)
        auction_daily["cumulative_revenue"] = auction_daily["daily_revenue_sum"].cumsum()
        auction_daily["cumulative_profit"] = auction_daily["cumulative_revenue"] - auction_cost
        auction_daily["time_since_auction"] = (auction_daily["sale_date"] - auction_daily["Auction_Date"]).dt.days
        auction_daily["auction_cost"] = auction_cost
        cumulative_rows.append(auction_daily[[
            "Auction_Date",
            "sale_date",
            "daily_revenue_sum",
            "cumulative_revenue",
            "cumulative_profit",
            "time_since_auction",
            "auction_cost",
        ]])

    if not cumulative_rows:
        return pd.DataFrame(columns=[
            "Auction_Date",
            "sale_date",
            "daily_revenue_sum",
            "cumulative_revenue",
            "cumulative_profit",
            "time_since_auction",
            "auction_cost",
        ])
    return pd.concat(cumulative_rows, ignore_index=True)


def build_auction_cumulative_profit_figure(df_cumulative_profit: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    sorted_auction_dates = sorted(df_cumulative_profit["Auction_Date"].dropna().unique())

    for index, auction_date in enumerate(sorted_auction_dates, start=1):
        df_auction = (
            df_cumulative_profit[df_cumulative_profit["Auction_Date"] == auction_date]
            .sort_values("time_since_auction")
        )
        if df_auction.empty:
            continue

        final_point = len(df_auction) - 1
        fig.add_trace(go.Scatter(
            x=df_auction["time_since_auction"],
            y=df_auction["cumulative_profit"],
            mode="lines+text",
            name=f"({index}) {auction_date.strftime('%Y-%m-%d')}",
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Days since auction: %{x}<br>"
                "Cumulative profit: %{y:$,.0f}<extra></extra>"
            ),
            text=[
                f"${value:,.0f}" if point_index == final_point else ""
                for point_index, value in enumerate(df_auction["cumulative_profit"])
            ],
            textposition="top right",
        ))

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=df_cumulative_profit["time_since_auction"].max(),
        y1=0,
        line=dict(color="Red", width=2, dash="dash"),
    )
    fig.update_layout(
        title={
            "text": "Interactive Cumulative Profit Over Time by Auction",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Days Since Auction",
        yaxis_title="Cumulative Profit ($)",
        hovermode="x unified",
        yaxis=dict(tickformat="$,.0f"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=120),
    )
    return fig


def cumulative_profit_axis_ranges(df_cumulative_profit: pd.DataFrame) -> tuple[list[float], list[float]]:
    max_days = df_cumulative_profit["time_since_auction"].max()
    min_profit = min(0, df_cumulative_profit["cumulative_profit"].min())
    max_profit = max(0, df_cumulative_profit["cumulative_profit"].max())
    profit_padding = max((max_profit - min_profit) * 0.05, 1)
    return [0, max_days], [min_profit - profit_padding, max_profit + profit_padding]


def build_single_auction_cumulative_profit_figure(
    df_cumulative_profit: pd.DataFrame,
    auction_date: pd.Timestamp,
    x_range: list[float],
    y_range: list[float],
    top_product_names: list[str] | None = None,
) -> go.Figure:
    df_auction = (
        df_cumulative_profit[df_cumulative_profit["Auction_Date"].dt.normalize() == auction_date.normalize()]
        .sort_values("time_since_auction")
    )
    fig = go.Figure()
    if df_auction.empty:
        return fig

    selected_auction_date = auction_date.normalize()
    comparison_dates = sorted(
        df_cumulative_profit["Auction_Date"].dropna().dt.normalize().unique()
    )
    for comparison_date in comparison_dates:
        if comparison_date == selected_auction_date:
            continue

        df_comparison = (
            df_cumulative_profit[df_cumulative_profit["Auction_Date"].dt.normalize() == comparison_date]
            .sort_values("time_since_auction")
        )
        if df_comparison.empty:
            continue

        fig.add_trace(go.Scatter(
            x=df_comparison["time_since_auction"],
            y=df_comparison["cumulative_profit"],
            mode="lines",
            name=comparison_date.strftime("%Y-%m-%d"),
            line=dict(color="rgba(150, 150, 150, 0.35)", width=1.5),
            hoverinfo="skip",
            showlegend=False,
        ))

    final_profit = df_auction["cumulative_profit"].iloc[-1]
    fig.add_trace(go.Scatter(
        x=df_auction["time_since_auction"],
        y=df_auction["cumulative_profit"],
        mode="lines+text",
        name=auction_date.strftime("%Y-%m-%d"),
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Days since auction: %{x}<br>"
            "Cumulative profit: %{y:$,.0f}<extra></extra>"
        ),
        text=[""] * (len(df_auction) - 1) + [f"${final_profit:,.0f}"],
        textposition="top right",
    ))
    fig.add_shape(
        type="line",
        x0=x_range[0],
        y0=0,
        x1=x_range[1],
        y1=0,
        line=dict(color="Red", width=2, dash="dash"),
    )
    if top_product_names:
        product_text = "<br>".join(
            f"{rank}. {escape(product_name)}"
            for rank, product_name in enumerate(top_product_names[:5], start=1)
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            align="left",
            text=f"<b>Top 5 by items sold</b><br>{product_text}",
            showarrow=False,
            bordercolor="#D9DEE8",
            borderwidth=1,
            borderpad=8,
            bgcolor="rgba(255, 255, 255, 0.88)",
            font=dict(size=12, color="#1F2937"),
        )
    fig.update_layout(
        title=f"Cumulative Profit: {auction_date:%Y-%m-%d}",
        xaxis_title="Days Since Auction",
        yaxis_title="Cumulative Profit ($)",
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range, tickformat="$,.0f"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=360,
        showlegend=False,
        margin=dict(t=60, b=40),
    )
    return fig


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


def daily_sales_performance(df: pd.DataFrame) -> dict:
    """Summarizes today's sales and rank among revenue-generating days."""
    sales = df.dropna(subset=["sale_date"]).copy()
    today = datetime.now(ZoneInfo("America/Denver")).date()

    empty_result = {
        "today": today,
        "sales_today": pd.DataFrame(),
        "total_revenue_today": 0,
        "items_sold_today": 0,
        "rank": None,
        "total_sales_days": 0,
        "percentile_rank": None,
    }
    if sales.empty:
        return empty_result

    sales["sale_date_only"] = sales["sale_date"].dt.date
    sales_today = sales[sales["sale_date_only"] == today].copy()
    total_revenue_today = sales_today["revenue"].sum()

    daily_revenue = (
        sales.groupby("sale_date_only", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    today_rank_row = daily_revenue[daily_revenue["sale_date_only"] == today]
    rank = int(today_rank_row.index[0] + 1) if not today_rank_row.empty else None
    total_sales_days = len(daily_revenue)
    percentile_rank = (rank / total_sales_days) * 100 if rank and total_sales_days else None

    return {
        "today": today,
        "sales_today": sales_today,
        "total_revenue_today": total_revenue_today,
        "items_sold_today": sales_today["revenue"].count(),
        "rank": rank,
        "total_sales_days": total_sales_days,
        "percentile_rank": percentile_rank,
    }


def yearly_tax_summary(df_sales: pd.DataFrame, df_donations: pd.DataFrame) -> pd.DataFrame:
    """Summarizes non-Venmo sales and tax-deductible donations by tax year."""
    summary_columns = [
        "Year",
        "Non-Venmo Sales",
        "Donation Deduction",
        "Taxable Total",
        "Taxable Sale Count",
        "Donation Count",
    ]

    if "venmo_flag" in df_sales.columns:
        sales = df_sales.dropna(subset=["sale_date"]).copy()
        sales["venmo_flag_numeric"] = pd.to_numeric(sales["venmo_flag"], errors="coerce")
        sales["revenue"] = pd.to_numeric(sales["revenue"], errors="coerce")
        taxable_sales = sales[sales["venmo_flag_numeric"].eq(0)].dropna(subset=["revenue"]).copy()
        if taxable_sales.empty:
            sales_summary = pd.DataFrame(columns=["Year", "Non-Venmo Sales", "Taxable Sale Count"])
        else:
            taxable_sales["Year"] = taxable_sales["sale_date"].dt.year
            sales_summary = (
                taxable_sales.groupby("Year", as_index=False)
                .agg(
                    **{
                        "Non-Venmo Sales": ("revenue", "sum"),
                        "Taxable Sale Count": ("revenue", "count"),
                    }
                )
            )
    else:
        sales_summary = pd.DataFrame(columns=["Year", "Non-Venmo Sales", "Taxable Sale Count"])

    if df_donations.empty:
        donation_summary = pd.DataFrame(columns=["Year", "Donation Deduction", "Donation Count"])
    else:
        donations = df_donations.dropna(subset=["Donated", "Value"]).copy()
        if donations.empty:
            donation_summary = pd.DataFrame(columns=["Year", "Donation Deduction", "Donation Count"])
        else:
            donations["Year"] = donations["Donated"].dt.year
            donation_summary = (
                donations.groupby("Year", as_index=False)
                .agg(
                    **{
                        "Donation Deduction": ("Value", "sum"),
                        "Donation Count": ("Value", "count"),
                    }
                )
            )

    summary = pd.merge(sales_summary, donation_summary, on="Year", how="outer")
    if summary.empty:
        return pd.DataFrame(columns=summary_columns)

    fill_values = {
        "Non-Venmo Sales": 0,
        "Donation Deduction": 0,
        "Taxable Sale Count": 0,
        "Donation Count": 0,
    }
    summary = summary.fillna(fill_values)
    summary["Year"] = summary["Year"].astype(int)
    summary["Taxable Sale Count"] = summary["Taxable Sale Count"].astype(int)
    summary["Donation Count"] = summary["Donation Count"].astype(int)
    summary["Taxable Total"] = summary["Non-Venmo Sales"] - summary["Donation Deduction"]
    return summary[summary_columns].sort_values("Year", ascending=False)


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
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=100),
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


def top_selling_product_names(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    name_column = next(
        (column for column in ["product", "Title", "product_categories", "product_category"] if column in df.columns),
        None,
    )
    output_columns = [
        "Auction Date",
        "Rank",
        "Product Name",
    ]
    if name_column is None or df.empty or "Auction_Date" not in df.columns:
        return pd.DataFrame(columns=output_columns)

    data = df.dropna(subset=["Auction_Date", name_column, "revenue"]).copy()
    data[name_column] = data[name_column].astype(str).str.strip()
    data = data[data[name_column].ne("")]
    if data.empty:
        return pd.DataFrame(columns=output_columns)

    summary = (
        data.groupby(["Auction_Date", name_column], as_index=False)
        .agg(
            **{
                "Items Sold": ("revenue", "count"),
                "Total Revenue": ("revenue", "sum"),
            }
        )
        .sort_values(["Auction_Date", "Items Sold", "Total Revenue"], ascending=[True, False, False])
    )
    summary["Rank"] = summary.groupby("Auction_Date").cumcount() + 1
    summary = summary[summary["Rank"].le(limit)].copy()
    summary["Auction Date"] = summary["Auction_Date"].dt.strftime("%Y-%m-%d")
    summary = summary.rename(columns={name_column: "Product Name"})
    return summary[output_columns]


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

donations_load_error = None
try:
    df_donations = import_donations_data()
except Exception as e:
    donations_load_error = e
    df_donations = pd.DataFrame()

listing_sale_records = listing_to_sale_records(filtered, df_listings)
avg_listing_to_sale_days = (
    listing_sale_records["days_to_sale"].mean() if not listing_sale_records.empty else np.nan
)
auction_listing_records = auction_to_listing_records(df_listings)
tax_summary = yearly_tax_summary(df, df_donations)

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

tab_overview, tab_auctions, tab_listings, tab_donations, tab_categories, tab_recent, tab_raw = st.tabs([
    "Overview", "By auction", "Listings", "Donations", "Categories", "Recent sales", "Raw data"
])

with tab_overview:
    today_sales = daily_sales_performance(df)
    st.subheader(f"Today's sales ({today_sales['today']:%Y-%m-%d})")
    if today_sales["total_revenue_today"] > 0:
        today_cols = st.columns(3)
        today_cols[0].metric("Today's Revenue", f"${today_sales['total_revenue_today']:,.0f}")
        today_cols[1].metric("Items Sold Today", f"{today_sales['items_sold_today']:,.0f}")
        if today_sales["rank"] is not None:
            today_cols[2].metric(
                "Daily Rank",
                f"#{today_sales['rank']} of {today_sales['total_sales_days']}",
                f"Top {today_sales['percentile_rank']:.0f}%",
            )
            st.caption(
                f"Today is the {today_sales['rank']} highest selling day, "
                f"ranking in the top {today_sales['percentile_rank']:.0f}% of sales days."
            )
        else:
            today_cols[2].metric("Daily Rank", "N/A")

        today_item_cols = [
            column for column in ["sale_date", "Title", "product", "product_category", "revenue", "Auction_Date"]
            if column in today_sales["sales_today"].columns
        ]
        st.dataframe(
            today_sales["sales_today"].sort_values("revenue", ascending=False)[today_item_cols].style.format({
                "revenue": "${:,.2f}",
            }),
            use_container_width=True,
        )
    else:
        st.info("No sales recorded for today.")

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

    st.subheader("Cumulative profit over time by auction")
    df_cumulative_profit = prepare_cumulative_profit_data(df, df_auction_cost)
    if not df_cumulative_profit.empty:
        excluded_dates = CUMULATIVE_PROFIT_EXCLUDED_DATES.normalize()
        auction_dates = df_cumulative_profit["Auction_Date"].dt.normalize()
        df_cumulative_profit = df_cumulative_profit[~auction_dates.isin(excluded_dates)].copy()

    if not df_cumulative_profit.empty:
        st.plotly_chart(build_auction_cumulative_profit_figure(df_cumulative_profit), use_container_width=True)
        cumulative_profit_x_range, cumulative_profit_y_range = cumulative_profit_axis_ranges(df_cumulative_profit)
    else:
        cumulative_profit_x_range, cumulative_profit_y_range = None, None
        st.info("No cumulative auction profit data found.")

    st.subheader("Auction charts with top 5 products")
    top_products = top_selling_product_names(filtered)
    if (
        cumulative_profit_x_range is not None
        and cumulative_profit_y_range is not None
        and not top_products.empty
    ):
        for auction_date, auction_products in top_products.groupby("Auction Date", sort=True):
            auction_timestamp = pd.to_datetime(auction_date, errors="coerce")
            if pd.isna(auction_timestamp):
                continue

            auction_profit = df_cumulative_profit[
                df_cumulative_profit["Auction_Date"].dt.normalize() == auction_timestamp.normalize()
            ]
            if auction_profit.empty:
                continue

            product_names = auction_products.sort_values("Rank")["Product Name"].tolist()
            st.plotly_chart(
                build_single_auction_cumulative_profit_figure(
                    df_cumulative_profit,
                    auction_timestamp,
                    cumulative_profit_x_range,
                    cumulative_profit_y_range,
                    product_names,
                ),
                use_container_width=True,
            )
    else:
        st.info("No product sales data found.")

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
    else:
        st.info("No sold listings with both a listing date and sale date were found for this sale date range.")

with tab_donations:
    st.subheader(f"{TAX_YEAR} tax summary")
    st.caption(f"Uses {TAX_YEAR} sales and donations only. Non-Venmo sales are rows where `venmo_flag` is 0.")

    if donations_load_error is not None:
        st.error("Could not load the Donations Google Sheet tab. Make sure the sheet is shared/public or accessible as CSV.")
        st.exception(donations_load_error)

    tax_year_summary = tax_summary[tax_summary["Year"].eq(TAX_YEAR)].copy()
    if not tax_year_summary.empty:
        totals = tax_year_summary.iloc[0]
        donation_cols = st.columns(3)
        donation_cols[0].metric("Non-Venmo Sales", f"${totals['Non-Venmo Sales']:,.0f}")
        donation_cols[1].metric("Donation Deduction", f"${totals['Donation Deduction']:,.0f}")
        donation_cols[2].metric("Taxable Total", f"${totals['Taxable Total']:,.0f}")

        chart_data = tax_year_summary.sort_values("Year")
        fig_tax = go.Figure()
        fig_tax.add_bar(x=chart_data["Year"], y=chart_data["Non-Venmo Sales"], name="Non-Venmo Sales")
        fig_tax.add_bar(x=chart_data["Year"], y=chart_data["Donation Deduction"], name="Donation Deduction")
        fig_tax.add_scatter(
            x=chart_data["Year"],
            y=chart_data["Taxable Total"],
            name="Taxable Total",
            mode="lines+markers+text",
            text=chart_data["Taxable Total"].map(lambda value: f"${value:,.0f}"),
            textposition="top center",
        )
        fig_tax.update_layout(
            barmode="group",
            xaxis_title="Year",
            yaxis_title="Dollars",
            yaxis=dict(tickformat="$,.0f"),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_tax, use_container_width=True)

        st.dataframe(
            tax_year_summary.style.format({
                "Non-Venmo Sales": "${:,.2f}",
                "Donation Deduction": "${:,.2f}",
                "Taxable Total": "${:,.2f}",
                "Taxable Sale Count": "{:,.0f}",
                "Donation Count": "{:,.0f}",
            }),
            use_container_width=True,
        )
    else:
        st.info(f"No {TAX_YEAR} donation or non-Venmo sales data found.")

    if not df_donations.empty:
        donation_detail_columns = [
            column for column in ["Donated", "Item", "Value", "Condition", "Received date", "Receved how"]
            if column in df_donations.columns
        ]
        donation_details = (
            df_donations.dropna(subset=["Donated", "Value"])
            .loc[lambda data: data["Donated"].dt.year.eq(TAX_YEAR)]
            .sort_values("Donated", ascending=False)[donation_detail_columns]
        )
        st.subheader("Donation records")
        st.dataframe(
            donation_details.style.format({
                "Value": "${:,.2f}",
                "Donated": format_date_for_display,
                "Received date": format_date_for_display,
            }),
            use_container_width=True,
        )

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
