# Data/ExtractData_IAModel/multivar_forecast.py
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers

from .config_iamodel import DATA_CSV_DIR, MODELS_DIR, ensure_dir


# ============================================================
# 1. DATA LOADING + PREPROCESSING
# ============================================================

def _load_raw_forecast_data(data_dir: Path = DATA_CSV_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load the daily regional CSVs for all gases + LST.

    Expected files in `data_dir`:
      - S5P_NO2_Tunisia_Regions_Daily_2024.csv
      - S5P_CO_Tunisia_Regions_Daily_2024.csv
      - S5P_CH4_Tunisia_Regions_Daily_2024.csv
      - S5P_O3_Tunisia_Regions_Daily_2024.csv
      - S5P_SO2_Tunisia_Regions_Daily_2024.csv
      - MODIS_LST_C_Tunisia_Regions_Daily_2019_2024.csv
    """
    no2_path = data_dir / "S5P_NO2_Tunisia_Regions_Daily_2024.csv"
    co_path  = data_dir / "S5P_CO_Tunisia_Regions_Daily_2024.csv"
    ch4_path = data_dir / "S5P_CH4_Tunisia_Regions_Daily_2024.csv"
    o3_path  = data_dir / "S5P_O3_Tunisia_Regions_Daily_2024.csv"
    so2_path = data_dir / "S5P_SO2_Tunisia_Regions_Daily_2024.csv"
    lst_path = data_dir / "MODIS_LST_C_Tunisia_Regions_Daily_2019_2024.csv"

    dfs = {
        "no2": pd.read_csv(no2_path),
        "co":  pd.read_csv(co_path),
        "ch4": pd.read_csv(ch4_path),
        "o3":  pd.read_csv(o3_path),
        "so2": pd.read_csv(so2_path),
        "lst": pd.read_csv(lst_path),
    }

    for df in dfs.values():
        df["date"] = pd.to_datetime(df["date"])

    return dfs


def _rename_mean(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Keep only (date, region, mean) and rename `mean` → desired column name."""
    df_local = df[["date", "region", "mean"]].copy()
    df_local["date"] = pd.to_datetime(df_local["date"])
    # IMPORTANT: Normalize region strings
    df_local["region"] = df_local["region"].astype(str).str.strip().str.lower()
    df_local = df_local.rename(columns={"mean": name})
    return df_local


def _clip_outliers(
    df: pd.DataFrame,
    cols,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    Simple quantile clipping for robustness.
    Values below lower_q or above upper_q are clipped to those quantiles.
    """
    df_clipped = df.copy()
    for c in cols:
        low = df_clipped[c].quantile(lower_q)
        high = df_clipped[c].quantile(upper_q)
        df_clipped[c] = df_clipped[c].clip(low, high)
    return df_clipped


def _build_master_df_forecast(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge NO2, CO, CH4, O3, SO2 and LST into a single time-series per region,
    interpolate missing values, clip outliers and add calendar + lag features.
    """
    no2_raw = dfs["no2"]
    co_raw  = dfs["co"]
    ch4_raw = dfs["ch4"]
    o3_raw  = dfs["o3"]
    so2_raw = dfs["so2"]
    lst_raw = dfs["lst"]

    merged_list = [
        _rename_mean(no2_raw, "NO2"),
        _rename_mean(co_raw,  "CO"),
        _rename_mean(ch4_raw, "CH4"),
        _rename_mean(o3_raw,  "O3"),
        _rename_mean(so2_raw, "SO2"),
        _rename_mean(lst_raw, "LST_C"),
    ]

    # Outer merge on (date, region)
    df_master_raw = merged_list[0]
    for df in merged_list[1:]:
        df_master_raw = pd.merge(df_master_raw, df, on=["date", "region"], how="outer")

    df_master_raw["date"] = pd.to_datetime(df_master_raw["date"])

    # We keep only 2019–2023 for training
    mask_year = df_master_raw["date"].dt.year.between(2019, 2023)
    df_master_raw = (
        df_master_raw.loc[mask_year]
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )

    # Time interpolation per region
    df_master = df_master_raw.copy().set_index("date")
    poll_cols = ["NO2", "CO", "CH4", "O3", "SO2", "LST_C"]

    df_interp = (
        df_master
        .groupby("region")[poll_cols]
        .apply(lambda g: g.interpolate(method="time").bfill().ffill())
    )
    df_interp = df_interp.reset_index(level=0, drop=True)
    df_master[poll_cols] = df_interp
    df_master = df_master.reset_index()

    # Outlier clipping
    df_master = _clip_outliers(df_master, poll_cols, 0.01, 0.99)

    # Calendar features
    df_master["date"] = pd.to_datetime(df_master["date"])
    df_master["year"]       = df_master["date"].dt.year.astype(int)
    df_master["month"]      = df_master["date"].dt.month.astype(int)
    df_master["dayofyear"]  = df_master["date"].dt.dayofyear.astype(int)
    df_master["dayofweek"]  = df_master["date"].dt.dayofweek.astype(int)
    df_master["season"]     = (df_master["month"] % 12 // 3 + 1).astype(int)
    df_master["is_weekend"] = df_master["dayofweek"].isin([5, 6]).astype(int)

    df_master["sin_doy"] = np.sin(2 * np.pi * df_master["dayofyear"] / 365.0)
    df_master["cos_doy"] = np.cos(2 * np.pi * df_master["dayofyear"] / 365.0)

    # Lag + rolling features per region
    for col in poll_cols:
        df_master[f"{col}_lag1"] = df_master.groupby("region")[col].shift(1)
        df_master[f"{col}_lag7"] = df_master.groupby("region")[col].shift(7)
        df_master[f"{col}_roll7_mean"] = (
            df_master.groupby("region")[col]
            .rolling(7)
            .mean()
            .reset_index(0, drop=True)
        )
        df_master[f"{col}_roll30_mean"] = (
            df_master.groupby("region")[col]
            .rolling(30)
            .mean()
            .reset_index(0, drop=True)
        )

    df_master = df_master.dropna()
    return df_master


# ============================================================
# 2. SEQUENCE BUILDER + TRAINING
# ============================================================

def _build_sequences_regression(
    df: pd.DataFrame,
    feature_cols,
    target_cols,
    seq_len: int,
):
    """
    Build sliding windows for all regions.

    For each region:
      - take sequences of `seq_len` days as X
      - next-day targets (all gases + LST) as y
    """
    X_list, y_list = [], []
    dates_list, regions_list = [], []

    for region in sorted(df["region"].unique()):
        df_r = df[df["region"] == region].sort_values("date").reset_index(drop=True)
        features = df_r[feature_cols].values
        targets = df_r[target_cols].values
        dates = df_r["date"].values

        for t in range(seq_len, len(df_r)):
            X_list.append(features[t - seq_len : t, :])
            y_list.append(targets[t])
            dates_list.append(dates[t])
            regions_list.append(region)

    X = np.stack(X_list)
    y = np.stack(y_list)
    return X, y, np.array(dates_list), np.array(regions_list)


def train_forecast_model(
    data_dir: Path = DATA_CSV_DIR,
    out_dir: Path | None = None,
    seq_len: int = 60,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    epochs: int = 30,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Train the multivariate LSTM forecaster (1-day ahead for all gases + LST).

    Saves:
      - model:   forecast_lstm.keras
      - scaler_X: forecast_X_scaler.pkl
      - scaler_y: forecast_Y_scaler.pkl
      - metadata: forecast_metadata.json
    """
    if out_dir is None:
        out_dir = MODELS_DIR / "forecast"
    ensure_dir(out_dir)

    print(f"[Forecast] Loading data from: {data_dir}")
    dfs = _load_raw_forecast_data(data_dir)
    df_master = _build_master_df_forecast(dfs)

    TARGET_COLS = ["NO2", "CO", "CH4", "O3", "SO2", "LST_C"]

    # use all numeric features except direct targets, date and region
    feature_cols = [
        c
        for c in df_master.columns
        if c not in ["date", "region"] + TARGET_COLS
    ]

    X, y, dates, _regions = _build_sequences_regression(
        df_master, feature_cols, TARGET_COLS, seq_len
    )

    # Time-based split
    order_idx = np.argsort(dates)
    X = X[order_idx]
    y = y[order_idx]
    dates = dates[order_idx]

    N = X.shape[0]
    n_train = int(train_frac * N)
    n_val = int(val_frac * N)
    n_test = N - n_train - n_val

    X_train_raw = X[:n_train]
    y_train_raw = y[:n_train]
    X_val_raw = X[n_train : n_train + n_val]
    y_val_raw = y[n_train : n_train + n_val]
    X_test_raw = X[n_train + n_val :]
    y_test_raw = y[n_train + n_val :]

    # --- Scaling ---
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    n_features = X_train_raw.shape[2]
    X_train_flat = X_train_raw.reshape(-1, n_features)
    X_val_flat = X_val_raw.reshape(-1, n_features)
    X_test_flat = X_test_raw.reshape(-1, n_features)

    X_train = scaler_X.fit_transform(X_train_flat).reshape(X_train_raw.shape)
    X_val = scaler_X.transform(X_val_flat).reshape(X_val_raw.shape)
    X_test = scaler_X.transform(X_test_flat).reshape(X_test_raw.shape)

    n_targets = len(TARGET_COLS)
    y_train_flat = y_train_raw.reshape(-1, n_targets)
    y_val_flat = y_val_raw.reshape(-1, n_targets)
    y_test_flat = y_test_raw.reshape(-1, n_targets)

    y_train = scaler_y.fit_transform(y_train_flat).reshape(y_train_raw.shape)
    y_val = scaler_y.transform(y_val_flat).reshape(y_val_raw.shape)
    y_test = scaler_y.transform(y_test_flat).reshape(y_test_raw.shape)

    # --- Model definition ---
    n_timesteps = X_train.shape[1]

    model = Sequential(
        [
            Input(shape=(n_timesteps, n_features)),
            LSTM(
                64,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(1e-3),
            ),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(32, kernel_regularizer=regularizers.l2(1e-3)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-3)),
            Dropout(0.2),
            Dense(n_targets),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae", RootMeanSquaredError()],
    )

    model_path = out_dir / "forecast_lstm.keras"
    checkpoint = ModelCheckpoint(
        model_path, monitor="val_loss", save_best_only=True, verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    # --- Evaluation on original scale ---
    y_test_pred_scaled = model.predict(X_test, verbose=0)
    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_scaled.reshape(-1, n_targets)
    )
    y_test_true = y_test_raw.reshape(-1, n_targets)

    mae = mean_absolute_error(y_test_true, y_test_pred, multioutput="raw_values")
    mse = mean_squared_error(y_test_true, y_test_pred, multioutput="raw_values")
    r2  = r2_score(y_test_true, y_test_pred, multioutput="raw_values")

    print("[Forecast] MAE per target:", dict(zip(TARGET_COLS, mae)))
    print("[Forecast] R2  per target:", dict(zip(TARGET_COLS, r2)))

    # --- Save artifacts ---
    with open(out_dir / "forecast_X_scaler.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(out_dir / "forecast_Y_scaler.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    metadata = {
        "seq_len": int(seq_len),
        "n_features": int(n_features),
        "feature_names": feature_cols,
        "target_names": TARGET_COLS,
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
    }
    with open(out_dir / "forecast_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "mae": {k: float(v) for k, v in zip(TARGET_COLS, mae)},
        "r2": {k: float(v) for k, v in zip(TARGET_COLS, r2)},
        "model_path": str(model_path),
    }


# ============================================================
# 3. INFERENCE HELPERS
# ============================================================

def load_forecast_artifacts(models_dir: Path | None = None):
    """
    Load trained LSTM + scalers + metadata from disk.
    """
    if models_dir is None:
        models_dir = MODELS_DIR / "forecast"

    model = tf.keras.models.load_model(models_dir / "forecast_lstm.keras")
    with open(models_dir / "forecast_X_scaler.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(models_dir / "forecast_Y_scaler.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    with open(models_dir / "forecast_metadata.json", "r") as f:
        metadata = json.load(f)

    return model, scaler_X, scaler_y, metadata


def make_input_window(
    df: pd.DataFrame,
    region: str,
    target_date,
    seq_len: int,
    feature_cols,
):
    """
    Build the last `seq_len`-day window for one region up to `target_date`.

    If `target_date` is not present for that region, we fall back to
    the latest available date in that region.

    Returns
    -------
    X_win : np.ndarray
        Array of shape (1, seq_len, n_features)
    used_date : pd.Timestamp
        The date actually used as "last observation" (target_date or fallback).
    """
    target_date = pd.to_datetime(target_date)

    # IMPORTANT: reset_index(drop=True) to have 0..N-1 positional index
    sub = (
        df[df["region"] == region]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if sub.empty:
        raise ValueError(f"No data for region '{region}'")

    # If the requested date is not available, use the latest for that region
    if target_date not in sub["date"].values:
        actual_date = sub["date"].max()
        print(
            f"Warning: {target_date.date()} not found for {region} → "
            f"using latest date {actual_date.date()}"
        )
        target_date = actual_date

    # Now index is positional (0..len-1), so this is safe:
    end_pos_array = np.where(sub["date"].values == target_date)[0]
    if len(end_pos_array) == 0:
        raise ValueError(
            f"Target date {target_date.date()} not found for region '{region}' "
            f"even after fallback."
        )
    end_pos = int(end_pos_array[0])
    start_pos = end_pos - seq_len + 1

    if start_pos < 0:
        raise ValueError(
            f"Not enough history ({seq_len} days) for {region} up to {target_date.date()}"
        )

    window = sub.loc[start_pos : end_pos, feature_cols].values

    if window.shape[0] != seq_len:
        raise RuntimeError(
            f"Expected {seq_len} rows in window, got {window.shape[0]} "
            f"for region '{region}' up to {target_date.date()}"
        )

    return window.reshape(1, seq_len, -1), target_date


def forecast_next_day(
    df_master: pd.DataFrame,
    region: str,
    last_obs_date: str,
    model,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    metadata: dict,
) -> pd.DataFrame:
    """
    Given master DF + loaded model/scalers, predict the next-day values.
    """
    seq_len = metadata["seq_len"]
    feature_cols = metadata["feature_names"]
    target_cols = metadata["target_names"]

    X_win, used_date = make_input_window(
        df_master, region, last_obs_date, seq_len, feature_cols
    )
    X_scaled = scaler_X.transform(
        X_win.reshape(-1, X_win.shape[-1])
    ).reshape(1, seq_len, -1)

    pred_scaled = model.predict(X_scaled, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled).flatten()

    forecast_date = pd.to_datetime(used_date) + pd.Timedelta(days=1)
    result = pd.DataFrame([pred], columns=target_cols)
    result["region"] = region
    result["date"] = forecast_date
    return result


# ============================================================
# 4. SERVICE WRAPPER FOR DJANGO
# ============================================================

_FORECAST_CONTEXT: Tuple[pd.DataFrame, Any, StandardScaler, StandardScaler, dict] | None = None


def get_forecast_context(reload: bool = False):
    """
    Load (or reuse) the full forecasting context:
      - df_master: merged + feature-engineered dataframe
      - model, scaler_X, scaler_y, metadata
    Used by the Django API layer.
    """
    global _FORECAST_CONTEXT

    if _FORECAST_CONTEXT is None or reload:
        dfs = _load_raw_forecast_data(DATA_CSV_DIR)
        df_master = _build_master_df_forecast(dfs)
        model, scaler_X, scaler_y, metadata = load_forecast_artifacts()
        _FORECAST_CONTEXT = (df_master, model, scaler_X, scaler_y, metadata)

    return _FORECAST_CONTEXT


def forecast_region_next_day(
    region: str,
    last_date: str | None = None,
    reload: bool = False,
) -> pd.DataFrame:
    """
    High-level function used by Django:
      - region: e.g. 'ariana'
      - last_date: 'YYYY-MM-DD' or None → use latest
    """
    df_master, model, scaler_X, scaler_y, metadata = get_forecast_context(
        reload=reload
    )

    if last_date is None:
        sub = df_master[df_master["region"] == region]
        if sub.empty:
            raise ValueError(f"No data for region '{region}'")
        last_date = sub["date"].max().strftime("%Y-%m-%d")

    return forecast_next_day(
        df_master=df_master,
        region=region,
        last_obs_date=last_date,
        model=model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        metadata=metadata,
    )


def forecast_region_next_day_dict(
    region: str,
    last_date: str | None = None,
    reload: bool = False,
) -> Dict[str, Any]:
    """
    Same as `forecast_region_next_day` but returns a JSON-ready dict.
    """
    df = forecast_region_next_day(region=region, last_date=last_date, reload=reload)
    row = df.iloc[0].copy()

    row["date"] = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
    for col in ["NO2", "CO", "CH4", "O3", "SO2", "LST_C"]:
        row[col] = float(row[col])

    return dict(row)
