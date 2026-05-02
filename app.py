import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from pathlib import Path


def resolve_logo_path() -> Path | None:
    explicit_candidates = [
        Path("assets/boxland2.png"),
        Path("boxland2.png"),
        Path("assets/boxland.png"),
        Path("boxland.png"),
        Path("assets/logo.png"),
        Path("logo.png"),
    ]
    for candidate in explicit_candidates:
        if candidate.exists():
            return candidate

    # Fallback: pick a likely image in root or assets even if filename casing differs.
    search_dirs = [Path("."), Path("assets")]
    for directory in search_dirs:
        if directory.exists():
            for file_path in directory.iterdir():
                if not file_path.is_file():
                    continue
                suffix = file_path.suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                stem = file_path.stem.lower()
                if "boxland" in stem or "logo" in stem:
                    return file_path
    return None


st.set_page_config(page_title="Auction Dashboard", layout="wide")

logo_path = resolve_logo_path()
if logo_path:
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(str(logo_path), use_container_width=True)
else:
    st.caption("Logo not found. Add `boxland.png` to repo root or `assets/boxland.png`.")

SHEET_ID = "1UkYDjeRaJlu3ByJYLeKzBScu7OwY6vxd2BgU-EnT41I"
GID = "900908138"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}"

AUCTION_COSTS = {
    "Auction_Date": [
        "8/19/2025", "8/26/2025", "9/11/2025", "9/18/2025", "9/25/2025", "10/2/2025",
        "10/17/2025", "10/23/2025", "12/11/2025", "12/23/2025", "1/8/2026", "1/22/2026",
        "1/29/2026", "2/2/2026", "2/12/2026", "3/5/2026", "3/19/2026", "4/30/2026"
    ],
    "auction_cost": [1800, 2000, 3386, 900, 650, 3367, 2811, 544.64, 2600, 1669, 3443, 1141, 849, 0, 1420, 761, 2647, 820],
}


def money_to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


@st.cache_data(ttl=3600)
def import_auction_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads Marketplace sales data from Google Sheets and merges in fixed auction costs."""
    df_google_sheet = pd.read_csv(CSV_URL)
    df_auction_cost = pd.DataFrame(AUCTION_COSTS)

    df_google_sheet["Auction_Date"] = pd.to_datetime(df_google_sheet["Auction_Date"], format="%m/%d/%Y", errors="coerce")
    df_google_sheet["sale_date"] = pd.to_datetime(df_google_sheet["sale_date"], format="%m/%d/%Y", errors="coerce")
    df_google_sheet["revenue"] = money_to_number(df_google_sheet["revenue"])

    df_auction_cost["Auction_Date"] = pd.to_datetime(df_auction_cost["Auction_Date"], format="%m/%d/%Y", errors="coerce")

    merged_df = pd.merge(df_google_sheet, df_auction_cost, on="Auction_Date", how="left")
    return merged_df, df_auction_cost


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
    return summary.sort_values("Auction_Date")


def daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.dropna(subset=["sale_date"]).groupby("sale_date", as_index=False).agg(
        revenue=("revenue", "sum"),
        items_sold=("revenue", "count"),
    )
    return daily.sort_values("sale_date")


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

from pathlib import Path

logo_path = Path("assets/boxland.png")
if logo_path.exists():
    st.image(str(logo_path), width=320)

st.caption("Auto-refreshes from Google Sheets. Data is cached for 1 hour.")

with st.sidebar:
    st.header("Filters")
    refresh = st.button("Refresh now")
    if refresh:
        st.cache_data.clear()

try:
    df, df_auction_cost = import_auction_data()
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
daily = daily_sales(filtered)

sold_items = filtered[filtered["sale_date"].notna()].copy()
total_revenue = sold_items["revenue"].sum()
total_cost = auction_summary["auction_cost"].sum()
total_profit = total_revenue - total_cost
items_sold = sold_items["revenue"].count()
avg_sale_price = sold_items["revenue"].mean() if items_sold else 0
profit_to_cost = total_profit / total_cost if total_cost else np.nan

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Revenue", f"${total_revenue:,.0f}")
c2.metric("Auction Cost", f"${total_cost:,.0f}")
c3.metric("Profit", f"${total_profit:,.0f}")
c4.metric("Items Sold", f"{items_sold:,.0f}")
c5.metric("Profit / Cost", "N/A" if np.isnan(profit_to_cost) else f"{profit_to_cost:.1%}")

st.divider()

tab_overview, tab_auctions, tab_categories, tab_recent, tab_raw = st.tabs([
    "Overview", "By auction", "Categories", "Recent sales", "Raw data"
])

with tab_overview:
    st.subheader("Sales over time")
    if not daily.empty:
        fig = px.line(daily, x="sale_date", y="revenue", markers=True, title="Daily Revenue")
        fig.update_layout(yaxis_title="Revenue", xaxis_title="Sale Date")
        st.plotly_chart(fig, use_container_width=True)

        daily["cumulative_revenue"] = daily["revenue"].cumsum()
        fig2 = px.line(daily, x="sale_date", y="cumulative_revenue", markers=True, title="Cumulative Revenue")
        fig2.update_layout(yaxis_title="Cumulative Revenue", xaxis_title="Sale Date")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No sales found for this date range.")

    st.subheader("Top revenue days")
    st.dataframe(daily.sort_values("revenue", ascending=False).head(10), use_container_width=True)

with tab_auctions:
    st.subheader("Auction performance")
    fig = go.Figure()
    fig.add_bar(x=auction_summary["Auction_Date"], y=auction_summary["total_revenue"], name="Revenue")
    fig.add_bar(x=auction_summary["Auction_Date"], y=auction_summary["auction_cost"], name="Cost")
    fig.add_scatter(x=auction_summary["Auction_Date"], y=auction_summary["profit"], name="Profit", mode="lines+markers")
    fig.update_layout(barmode="group", xaxis_title="Auction Date", yaxis_title="Dollars")
    st.plotly_chart(fig, use_container_width=True)

    display_summary = auction_summary.copy()
    st.dataframe(
        display_summary.style.format({
            "total_revenue": "${:,.0f}",
            "auction_cost": "${:,.0f}",
            "profit": "${:,.0f}",
            "profit_to_cost": "{:.1%}",
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
