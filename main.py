
# scenario_builder_app.py
# Streamlit + Plotly app for time-series scenario building (not forecasting)
# Window-cutter semantics + multi-column export
#
# Run: streamlit run scenario_builder_app.py

import io
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_TITLE = "Time-Series Scenario Builder"


# ------------- Utilities -------------
def _detect_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    candidates = [c for c in df.columns if str(c).lower() in {"date", "data", "dt", "timestamp", "time"}]
    if candidates:
        return candidates[0]
    c0 = df.columns[0]
    try:
        pd.to_datetime(df[c0])
        return c0
    except Exception:
        return None


def _infer_freq(dt: pd.Series) -> Optional[str]:
    dt = pd.to_datetime(dt).sort_values().drop_duplicates()
    if len(dt) < 3:
        return None
    try:
        f = pd.infer_freq(dt)
        if f:
            return f
    except Exception:
        pass
    deltas = dt.diff().dropna()
    if len(deltas) == 0:
        return None
    med = deltas.median()
    days = med / pd.Timedelta(days=1)
    if abs(days - 1) < 0.01:
        return "D"
    if abs(days - 7) < 0.01:
        return "W"
    if 27 <= days <= 31:
        return "MS"
    if 89 <= days <= 92:
        return "QS"
    return None


def _resample_to_length(values: np.ndarray, target_len: int) -> np.ndarray:
    if len(values) == 0 or target_len <= 0:
        return np.array([], dtype=float)
    if len(values) == target_len:
        return values.astype(float)
    x_old = np.linspace(0, 1, num=len(values))
    x_new = np.linspace(0, 1, num=target_len)
    return np.interp(x_new, x_old, values).astype(float)


def _build_transformed_window(
    series: pd.Series,
    invert: bool,
    amp_scale: float,
    amp_offset: float,
    time_scale: float,
    cut_len: int,
) -> np.ndarray:
    """
    Window-cutter semantics:
    1) Take exactly `cut_len` points from the observation window (after optional invert).
    2) Apply amplitude transform.
    3) Time scaling so OUTPUT length becomes round(cut_len * time_scale).
       * time_scale > 1 => stretch (slower; more points)
         time_scale < 1 => compress (faster; fewer points)
    """
    arr_all = series.to_numpy(dtype=float)

    if invert:
        arr_all = arr_all[::-1]

    cut_len = max(1, int(cut_len))
    if len(arr_all) < cut_len:
        arr = arr_all.copy()
    else:
        arr = arr_all[:cut_len].copy()

    arr = arr.astype(float) * amp_scale + amp_offset

    new_len = max(1, int(round(cut_len * time_scale)))
    trans = _resample_to_length(arr, new_len)
    return trans.astype(float)


def _build_transition(start_val: float, end_val: float, length: int, method: str) -> np.ndarray:
    if length <= 0 or method == "none":
        return np.array([], dtype=float)
    if method == "flat":
        return np.full(length, start_val, dtype=float)
    return np.linspace(start_val, end_val, num=length + 1, dtype=float)[:-1]


# ------------- Streamlit App -------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Build future scenarios on an existing timeseries by selecting a past time, diminish/enlarging its amplitude and stretching/compressing its time frame to met your objectives.")

    st.header("Import")
    with st.expander("Import File",expanded=True):
        uploaded = st.file_uploader("Upload time series file", type=["csv", "parquet", "xlsx"])
        sep = st.text_input("Specify CSV separator", value=",")

    if uploaded is None:
        st.info("Upload a file with at least a date column and one numeric value column.")
        st.stop()

    fname = uploaded.name.lower()
    if fname.endswith(".csv"):
        df = pd.read_csv(uploaded, sep=sep)
    elif fname.endswith(".parquet"):
        df = pd.read_parquet(uploaded)
    else:
        df = pd.read_excel(uploaded)
        
    with st.expander("Select reference columns",expanded=True):

        date_col_default = _detect_date_col(df) or df.columns[0]
        date_col = st.selectbox("Date", options=list(df.columns), index=list(df.columns).index(date_col_default))

        numeric_cols = [c for c in df.columns if c != date_col and pd.to_numeric(df[c], errors="coerce").notna().any()]
        if not numeric_cols:
            st.error("No numeric columns found. Please provide at least one numeric value column.")
            st.stop()

        value_col = st.selectbox("Value column (for preview)", options=numeric_cols, index=0)
    
    df = df[[date_col] + numeric_cols].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[value_col])

    if df.empty:
        st.error("No valid rows after parsing dates and values.")
        st.stop()    
    
    
    st.header("Scenario Definition")    
    with st.expander("Observation Period",expanded=True):
        # Observation window
        min_dt, max_dt = df[date_col].min(), df[date_col].max()
        c1, c2 = st.columns(2)
        with c1:
            obs_start = st.date_input("Window start", value=min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
        with c2:
            obs_end = st.date_input("Window end", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())

        if pd.to_datetime(obs_start) > pd.to_datetime(obs_end):
            st.error("Observation window start must be <= end.")
            st.stop()

        mask = (df[date_col] >= pd.to_datetime(obs_start)) & (df[date_col] <= pd.to_datetime(obs_end))
        obs = df.loc[mask, [date_col, value_col]].copy()
        if obs.empty:
            st.error("Observation window has no data for the selected column.")
            st.stop()


    # Scenario settings
    with st.expander("Transformation Parameters",expanded=True):
        with st.caption("Time"):
            time_scale = st.number_input("Time scale (1=unchanged; >1 = slower/stretch; <1 = faster/compress)", min_value=0.01, value=1.0, step=0.05, format="%.3f")

        with st.caption("Value"):  
            c5, c6=st.columns(2)
            with c5:
                amp_scale = st.number_input("Amplitude scale (×)", min_value=0.0, value=1.0, step=0.1, format="%.3f")
            with c6:
                amp_offset = st.number_input("Amplitude offset (+)", value=0.0, step=0.1, format="%.3f")
    with st.expander("Transition Period",expanded=True):
        c6, c7 = st.columns(2)
        with c6:
            transition_len = st.number_input("Transition length (points)", min_value=0, value=0, step=1)
        with c7:
            transition_method = st.selectbox("Transition method", options=["none", "linear", "flat"], index=0)

    with st.expander("Output",expanded=True):
        inferred_freq = _infer_freq(df[date_col])
        freq_choice = st.selectbox("Frequency for future dates", options=[inferred_freq] + ["D", "W", "MS", "QS", "H"], index=0 if inferred_freq else 1, help=f"Inferred: {inferred_freq or 'None'}")
        last_hist_dt = df[date_col].max()
        step = pd.tseries.frequencies.to_offset(freq_choice) if freq_choice else pd.tseries.frequencies.to_offset("D")
        default_start = (last_hist_dt + step)
        scenario_start = st.date_input("Scenario start date", value=default_start.date())
        scenario_length = st.number_input("Cut length (points taken from observation window)", min_value=1, value=max(1, len(obs)), step=1)
        invert = st.checkbox("Reverse order (invert in time)", value=False)
        apply_all = st.checkbox("Apply transformation to ALL numeric columns (export)", value=True)
      
    


    # Build transformed (preview column)
    transformed = _build_transformed_window(
        obs[value_col],
        invert=invert,
        amp_scale=amp_scale,
        amp_offset=amp_offset,
        time_scale=time_scale,
        cut_len=int(scenario_length),
    )

    # Build future dates (shifted by transition length)
    scenario_start_ts = pd.to_datetime(scenario_start)
    step = pd.tseries.frequencies.to_offset(freq_choice) if freq_choice else pd.tseries.frequencies.to_offset("D")
    scenario_start_effective = scenario_start_ts + int(transition_len) * step

    out_len = int(len(transformed))
    if freq_choice:
        future_idx = pd.date_range(start=scenario_start_effective, periods=out_len, freq=freq_choice)
    else:
        future_idx = pd.date_range(start=scenario_start_effective, periods=out_len, freq="D")

    # Transition segment for preview column
    # (Transition spans exactly `transition_len` stamps ending one step before first scenario date)
    hist_series = pd.Series(df[value_col].to_numpy(dtype=float), index=pd.to_datetime(df[date_col]))
    hist_last_val = hist_series.iloc[-1]
    scen_first_val = float(transformed[0]) if len(transformed) else hist_last_val
    if transition_len > 0:
        last_trans_ts = future_idx[0] - step
        transition_idx = pd.date_range(end=last_trans_ts, periods=int(transition_len), freq=step)
    else:
        transition_idx = pd.DatetimeIndex([], dtype="datetime64[ns]")
    transition_vals = _build_transition(hist_last_val, scen_first_val, int(transition_len), transition_method)
    if len(transition_vals) != len(transition_idx):
        if len(transition_vals) > len(transition_idx):
            transition_vals = transition_vals[: len(transition_idx)]
        else:
            transition_vals = np.pad(transition_vals, (0, len(transition_idx) - len(transition_vals)), mode="edge") if len(transition_vals) > 0 else np.zeros(len(transition_idx), dtype=float)

    trans_series = pd.Series(transition_vals, index=transition_idx)
    scen_series = pd.Series(np.asarray(transformed, dtype=float), index=future_idx)
    combined = pd.concat([hist_series, trans_series, scen_series])

    # Plot
    st.header("Overview")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_series.index, y=hist_series.values, name="Historical", mode="lines"))
    if len(trans_series) > 0:
        fig.add_trace(go.Scatter(x=trans_series.index, y=trans_series.values, name=f"Transition ({transition_method})", mode="lines"))
    fig.add_trace(go.Scatter(x=scen_series.index, y=scen_series.values, name="Scenario", mode="lines"))

    if len(trans_series) > 0:
        fig.add_vrect(
            x0=trans_series.index.min(),
            x1=trans_series.index.max(),
            fillcolor="LightGrey",
            opacity=0.25,
            line_width=0,
            annotation_text="Transition",
            annotation_position="top left",
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title=value_col,
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Preview shows the selected column. If 'Apply to ALL numeric columns' is on, the same transforms are applied to every numeric column for export (with identical cut/time scaling and the same transition timing).")

    # --- Multi-column scenario build (optional export) ---
    all_cols_out = {}
    for col in (numeric_cols if apply_all else [value_col]):
        hist_series_col = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float), index=pd.to_datetime(df[date_col])).dropna()

        obs_col = df.loc[mask, [date_col, col]].copy()
        obs_col[col] = pd.to_numeric(obs_col[col], errors="coerce")
        obs_col = obs_col.dropna(subset=[col])
        if obs_col.empty:
            continue

        transformed_col = _build_transformed_window(
            obs_col[col],
            invert=invert,
            amp_scale=amp_scale,
            amp_offset=amp_offset,
            time_scale=time_scale,
            cut_len=int(scenario_length),
        )

        # Force same output length across columns
        if len(transformed_col) != out_len:
            transformed_col = _resample_to_length(np.asarray(transformed_col, dtype=float), out_len)

        # Per-column transition
        hist_last_val_col = hist_series_col.iloc[-1] if len(hist_series_col) else np.nan
        scen_first_val_col = float(transformed_col[0]) if len(transformed_col) else hist_last_val_col
        transition_vals_col = _build_transition(hist_last_val_col, scen_first_val_col, int(transition_len), transition_method)
        if len(transition_vals_col) != len(transition_idx):
            if len(transition_vals_col) > len(transition_idx):
                transition_vals_col = transition_vals_col[: len(transition_idx)]
            else:
                transition_vals_col = np.pad(transition_vals_col, (0, len(transition_idx) - len(transition_vals_col)), mode="edge") if len(transition_vals_col) > 0 else np.zeros(len(transition_idx), dtype=float)

        trans_series_col = pd.Series(transition_vals_col, index=transition_idx)
        scen_series_col = pd.Series(np.asarray(transformed_col, dtype=float), index=future_idx)
        combined_col = pd.concat([hist_series_col, trans_series_col, scen_series_col])
        all_cols_out[col] = combined_col

    if all_cols_out:
        combined_df = pd.DataFrame(all_cols_out).sort_index()
    else:
        combined_df = pd.DataFrame({value_col: combined}).sort_index()

    # Export
    out_df = combined_df.reset_index().rename(columns={"index": "date"})
    st.dataframe(out_df.tail(20), use_container_width=True)

    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download ALL columns (historical + transition + scenario)", csv, file_name="scenario_built_all_columns.csv", mime="text/csv")


if __name__ == "__main__":
    main()
