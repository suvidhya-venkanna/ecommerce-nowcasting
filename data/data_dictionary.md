# Data Dictionary

Describes every column in `data/processed/full_dataset_clean.csv`, the merged, model-ready dataset
produced by `notebooks/01_data_pipeline_eda.ipynb`.

## Publication Lead Time (critical for nowcasting)

Every source becomes available at a **different point in time relative to the reference month
(T)** it describes. This asymmetry is the entire reason a nowcasting approach is useful here, so it
is documented explicitly rather than left implicit, since it determines what information the model
is actually allowed to use at prediction time.

## Data Sources and Acquisition

| Data Source | Description | Source URL |
|---|---|---|
| **Eurostat Turnover** | Monthly index of retail trade turnover (NACE G47.91). Search for **sts_trtu_m** in the homepage | https://ec.europa.eu/eurostat/databrowser/view/sts_trtu_m |
| **Eurostat HICP** | Harmonised Index of Consumer Prices (monthly). Search for **prc_hicp_minr** in the homepage  | https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_minr |
| **Google Trends** | Relative search interest **("online shopping", NL**) | https://trends.google.com/trends/ |

| Source | Typical Publication Lag | Available at Nowcast Time (T+0)? |
|---|---|---|
| `ecommerce_turnover` (target, Eurostat `sts_trtu_m`, NACE G47.91, Netherlands) | ~5-10 weeks after month-end | **No** - this is precisely the value being nowcast; it does not exist yet at T+0 |
| `hicp_inflation` (Eurostat `prc_hicp_manr`) | Flash estimate within days of month-end; full release ~2-3 weeks after | Partially - a flash estimate is close to real-time, but this project uses the published (non-flash) monthly value, which lags slightly |
| `search_index` (Google Trends, "online shopping", geo=NL) | Same day / near real-time | **Yes** - fully available at T+0, which is what gives it nowcasting value |

**Practical consequence:** at the moment a nowcast for month T is actually needed, the model can
only use the current month's `search_index` (used un-lagged - this is its main value as a leading
indicator), but must rely on *lagged* (month T-1 or earlier) values of `hicp_inflation` and of the
turnover target itself. This is why `ecommerce_turnover_lag1/3/12` and `search_index_lag1/3/12` are
engineered as explicitly lagged features, while raw `search_index` is used at lag 0.

## Column-Level Detail

| Column | Meaning | Type | Units / Format | Allowed Values | Target Variable | Key Assumptions |
|---|---|---|---|---|---|---|
| `Date` | Reference month for the observation (DataFrame index) | Datetime (index) | `YYYY-MM-01` (month start) | 2018-01 onward | No | Restricted to Google Trends' supported range so all sources are genuinely comparable (see below) |
| `ecommerce_turnover` | Turnover for retail sale via mail order houses / internet, Netherlands (NACE Rev. 2 class G47.91), seasonally and calendar adjusted | Continuous (float) | Index (Eurostat `s_adj = SCA` series) | Varies by period; check `data/raw/eurostat_turnover_G4791_NL_raw.csv` for the real observed range | **Yes (target)** | Seasonally and calendar adjusted (SCA), so calendar/seasonal effects are already partly smoothed by Eurostat before this project sees the data - this differs from a not-seasonally-adjusted (NSA) series and should be kept in mind when interpreting any seasonality found in EDA |
| `hicp_inflation` | Harmonised Index of Consumer Prices, annual rate of change, Netherlands (COICOP `CP00`, all-items) | Continuous (float) | % change year-over-year | Typically -2% to +12% depending on period | No (feature) | Used at its own natural monthly lag; not assumed to have any specific lead/lag relationship with turnover - that is left to be tested empirically |
| `search_index` | Google Trends relative search interest for "online shopping", Netherlands | Continuous (float) | 0-100 (Google's relative-interest normalization within the queried date range/region) | 0 to 100 | No (feature) | Values are *relative*, not an absolute search-volume count, and are only comparable within a single query using a fixed, unchanged date range and region (`START_DATE` to `TODAY`, `geo='NL'`) - re-running the query over a different window changes the normalization |
| `ecommerce_turnover_lag1` / `lag3` / `lag12` | `ecommerce_turnover` shifted 1, 3, or 12 months into the past | Continuous (float) | Same as `ecommerce_turnover` | Same as `ecommerce_turnover` | No (engineered feature) | Built via `.shift()` only (past values), never future values, to avoid lookahead leakage; gives the model legitimate autoregressive information |
| `search_index_lag1` / `lag3` / `lag12` | `search_index` shifted 1, 3, or 12 months into the past | Continuous (float) | Same as `search_index` | 0 to 100 | No (engineered feature) | Captures momentum/trend in search interest; the un-lagged `search_index` column remains the primary nowcasting signal, since it is the only feature genuinely available at T+0 |
| `month` | Calendar month of the observation | Categorical (int) | 1-12 | 1, 2, ..., 12 | No (feature) | Included as a simple seasonal indicator for tree-based/ML models (XGBoost), which cannot infer cyclical calendar effects from a date index alone the way ARIMA can |

## Additional Notes and Assumptions

- **Geographic scope:** all series are for the Netherlands (`geo = 'NL'`) only. This project does
  not currently include an EU-wide comparison.
- **Date range:** restricted to `START_DATE` (2018-01-01) onward, even though Eurostat's turnover
  and HICP series both go back to the early 1990s. This is a deliberate fix (see the notebook's
  intro cell for details) - Google Trends only supports a limited historical window, and an earlier
  version of this pipeline let Eurostat's much longer history leak into the merged dataset, which
  caused decades of `search_index` values to be back-filled with a single real data point rather
  than genuine historical search interest. Restricting the whole dataset to the window where all
  three sources have real, non-fabricated data was judged more defensible than keeping more rows at
  the cost of data integrity.
- **Missing value handling:** linear interpolation is applied first (`df.interpolate(method='linear')`),
  followed by forward-fill and back-fill for any remaining edge gaps. Because the date range is now
  restricted to `START_DATE` onward (see above), remaining gaps should be small, isolated reporting
  gaps rather than large structural ones - this should still be visually confirmed in the EDA
  (e.g., a missing-value count and a plot of any interpolated stretches) before the final submission.
- **Train/test split (for modeling, not yet implemented in this notebook):** any model built on this
  dataset must use a strict time-based split (e.g., train on 2018-2023, test on 2024-2025) - the
  dataset must never be shuffled, since shuffling would leak future information into training for a
  time-series problem. `TimeSeriesSplit` cross-validation (multiple rolling folds) is recommended
  over a single split for the final model evaluation.
- **No personally identifiable information (PII):** all data is aggregate, publicly published
  macroeconomic and search-trend data. No individual-level or transaction-level data is used.
