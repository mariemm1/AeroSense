# Data/ExtractData_IAModel/aqi_service.py
from __future__ import annotations

import numpy as np

from .aqi_classification import (
    load_aqi_artifacts,
    classify_sequence,
    _load_raw_aqi_data,
    _build_master_df_classification,
)
from .config_iamodel import DATA_CSV_DIR

# Global cache for model + scaler + metadata
_AQI_CONTEXT = None  # (model, scaler, metadata)


def get_aqi_context(reload: bool = False):
    """
    Lazy-load AQI model + scaler + metadata.
    Avoids crashing Django if artifacts are not trained yet.
    """
    global _AQI_CONTEXT

    if _AQI_CONTEXT is None or reload:
        model, scaler, metadata = load_aqi_artifacts()
        _AQI_CONTEXT = (model, scaler, metadata)

    return _AQI_CONTEXT


def classify_region_latest_window(region: str, reload: bool = False) -> dict:
    """
    High-level helper for the API:

      - loads raw CSVs
      - rebuilds master dataframe
      - takes last seq_len days for the given region
      - calls classify_sequence(...)
      - returns a JSON-ready dict
    """
    model, scaler, metadata = get_aqi_context(reload=reload)

    dfs = _load_raw_aqi_data(DATA_CSV_DIR)
    df_master = _build_master_df_classification(dfs)

    feature_cols = metadata["feature_names"]
    seq_len = metadata["seq_len"]

    df_r = df_master[df_master["region"] == region].sort_values("date")

    if len(df_r) < seq_len:
        raise ValueError(
            f"Not enough days for region '{region}': "
            f"need {seq_len}, have {len(df_r)}"
        )

    seq = df_r[feature_cols].values[-seq_len:]  # (seq_len, n_features)

    cls, proba = classify_sequence(seq, model, scaler, metadata)

    class_names = metadata.get("class_names", {})
    if isinstance(class_names, dict):
        class_name = class_names.get(str(cls), class_names.get(cls, str(cls)))
    else:
        class_name = str(cls)

    last_date = df_r["date"].max()

    return {
        "region": region,
        "date": last_date.strftime("%Y-%m-%d"),
        "class_id": int(cls),
        "class_name": class_name,  # "Good" / "Moderate" / "Unhealthy"
        "probabilities": {
            "0": float(proba[0]),
            "1": float(proba[1]),
            "2": float(proba[2]),
        },
    }

# ADD THIS near classify_region_latest_window

def classify_forecast_for_region(
    region: str,
    forecast_dict: dict,
    reload: bool = False,
) -> dict:
    """
    Take the multivariate forecast (NO2, CO, CH4, O3, SO2, LST_C, date)
    and classify the AIR QUALITY of that *forecasted day* using
    the same LSTM AQI classifier.

    Logic:
      - load df_master (same as training/classification)
      - take last (seq_len - 1) days for this region
      - build a fake row for forecast_date with forecasted gases + LST
        and recomputed gas_mean, gas_sum, sin_doy, cos_doy
      - stack them into one sequence of length seq_len
      - run classify_sequence(...)
    """
    model, scaler, metadata = get_aqi_context(reload=reload)

    # rebuild the same master dataframe
    dfs = _load_raw_aqi_data(DATA_CSV_DIR)
    df_master = _build_master_df_classification(dfs)

    feature_cols = metadata["feature_names"]
    seq_len = metadata["seq_len"]

    df_r = df_master[df_master["region"] == region].sort_values("date").copy()
    if len(df_r) < (seq_len - 1):
        raise ValueError(
            f"Not enough history for region '{region}' to classify forecast: "
            f"need at least {seq_len-1}, have {len(df_r)}"
        )

    # --- build last (seq_len - 1) real days ---
    hist = df_r.iloc[-(seq_len - 1) :].copy()

    # --- build forecast row with same feature schema ---
    import pandas as pd
    from datetime import datetime

    # parse forecast date
    f_date = pd.to_datetime(forecast_dict["date"]).normalize()

    # base columns
    row = {
        "date": f_date,
        "region": region,
        "NO2": float(forecast_dict["NO2"]),
        "CO": float(forecast_dict["CO"]),
        "CH4": float(forecast_dict["CH4"]),
        "O3": float(forecast_dict["O3"]),
        "SO2": float(forecast_dict["SO2"]),
        "LST_C": float(forecast_dict["LST_C"]),
    }

    # gas_mean / gas_sum same as in _build_master_df_classification
    poll_cols = ["NO2", "CO", "CH4", "O3", "SO2"]
    gases_vals = np.array([row[c] for c in poll_cols], dtype=float)
    row["gas_mean"] = float(gases_vals.mean())
    row["gas_sum"] = float(gases_vals.sum())

    # time features
    dayofyear = int(f_date.dayofyear)
    row["sin_doy"] = float(np.sin(2 * np.pi * dayofyear / 365.0))
    row["cos_doy"] = float(np.cos(2 * np.pi * dayofyear / 365.0))

    # (we skip year/month/dayofweek/season/is_weekend because they
    # are not in FEATURE_COLS used for the LSTM classifier; if they
    # are in feature_names, you can also add them here similarly.)

    forecast_row = {k: row[k] for k in feature_cols}

    # build the full sequence (seq_len, n_features)
    hist_features = hist[feature_cols].values  # (seq_len-1, n_features)
    last_features = np.array([list(forecast_row.values())], dtype=float)  # (1, n_features)
    seq = np.vstack([hist_features, last_features])  # (seq_len, n_features)

    # classify this sequence
    cls, proba = classify_sequence(seq, model, scaler, metadata)

    class_names = metadata.get("class_names", {})
    if isinstance(class_names, dict):
        class_name = class_names.get(str(cls), class_names.get(cls, str(cls)))
    else:
        class_name = str(cls)

    return {
        "class_id": int(cls),
        "class_name": class_name,
        "probabilities": {
            "0": float(proba[0]),
            "1": float(proba[1]),
            "2": float(proba[2]),
        },
    }
