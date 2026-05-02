# Auction / Facebook Marketplace Dashboard

This converts the Google Colab notebook into a lightweight Streamlit dashboard.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free on Streamlit Community Cloud

1. Create a GitHub repo.
2. Upload `app.py` and `requirements.txt`.
3. Go to Streamlit Community Cloud.
4. Connect the GitHub repo.
5. Set the main file to `app.py`.
6. Deploy.

## Auto-refresh behavior

The dashboard reads from Google Sheets and caches the result for 1 hour:

```python
@st.cache_data(ttl=3600)
```

You can change the refresh interval by editing the `ttl` value in `app.py`.

## Notes

The dashboard intentionally keeps only the useful high-level pieces:

- revenue, cost, profit, item count, profit-to-cost
- daily revenue and cumulative revenue
- auction-level performance
- category-level performance
- recent sales
- raw data for debugging

It does not include every chart from the original Colab notebook.
