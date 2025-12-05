# Data/ExtractData_IAModel/aqi_classification.py
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from .config_iamodel import DATA_CSV_DIR, MODELS_DIR, ensure_dir


# -------------------------------------------------------------------
# 1. Data loading
# -------------------------------------------------------------------

def _load_raw_aqi_data(data_dir: Path = DATA_CSV_DIR):
    """
    Load all CSVs needed for AQI classification.
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
    """
    Helper to select (date, region, mean) and rename 'mean' to a pollutant name.
    """
    df_local = df[["date", "region", "mean"]].copy()
    df_local["date"] = pd.to_datetime(df_local["date"])
    df_local["region"] = df_local["region"].astype(str).str.strip().str.lower()
    df_local = df_local.rename(columns={"mean": name})
    return df_local


def _clip_outliers(
    df: pd.DataFrame,
    cols,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
):
    """
    Clip outliers in specified columns using quantiles.
    """
    df_clipped = df.copy()
    for c in cols:
        low = df_clipped[c].quantile(lower_q)
        high = df_clipped[c].quantile(upper_q)
        df_clipped[c] = df_clipped[c].clip(low, high)
    return df_clipped


def _build_master_df_classification(dfs: dict) -> pd.DataFrame:
    """
    Merge gases + LST, interpolate, clip outliers, create time-based features,
    and derive AQI_class (0, 1, 2) based on quantiles of gas_mean.
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

    df_master_raw = merged_list[0]
    for df in merged_list[1:]:
        df_master_raw = pd.merge(df_master_raw, df, on=["date", "region"], how="outer")

    df_master_raw["date"] = pd.to_datetime(df_master_raw["date"])

    # Restrict to 2019–2022 as in your original notebook
    mask_year = df_master_raw["date"].dt.year.between(2019, 2022)
    df_master_raw = (
        df_master_raw.loc[mask_year]
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )

    # Interpolation per region
    df_master = df_master_raw.copy().set_index("date")
    cols_to_interp = ["NO2", "CO", "CH4", "O3", "SO2", "LST_C"]

    df_interp = (
        df_master
        .groupby("region")[cols_to_interp]
        .apply(lambda g: g.interpolate(method="time").bfill().ffill())
    )
    df_interp = df_interp.reset_index(level=0, drop=True)
    df_master[cols_to_interp] = df_interp
    df_master = df_master.reset_index()

    # Outlier clipping
    df_master = _clip_outliers(
        df_master,
        cols=cols_to_interp,
        lower_q=0.01,
        upper_q=0.99,
    )

    # Time-based and cyclic features
    df_master["date"] = pd.to_datetime(df_master["date"])
    df_master["year"]       = df_master["date"].dt.year.astype(int)
    df_master["month"]      = df_master["date"].dt.month.astype(int)
    df_master["dayofyear"]  = df_master["date"].dt.dayofyear.astype(int)
    df_master["dayofweek"]  = df_master["date"].dt.dayofweek.astype(int)
    df_master["season"]     = (df_master["month"] % 12 // 3 + 1).astype(int)
    df_master["is_weekend"] = df_master["dayofweek"].isin([5, 6]).astype(int)

    df_master["sin_doy"] = np.sin(2 * np.pi * df_master["dayofyear"] / 365.0)
    df_master["cos_doy"] = np.cos(2 * np.pi * df_master["dayofyear"] / 365.0)

    poll_cols = ["NO2", "CO", "CH4", "O3", "SO2"]
    df_master["gas_mean"] = df_master[poll_cols].mean(axis=1)
    df_master["gas_sum"]  = df_master[poll_cols].sum(axis=1)

    # AQI_raw + quantiles → AQI_class
    df_master["AQI_raw"] = df_master["gas_mean"]
    q1 = df_master["AQI_raw"].quantile(1 / 3)
    q2 = df_master["AQI_raw"].quantile(2 / 3)

    def aqi_class_from_raw(x: float) -> int:
        if x <= q1:
            return 0  # Good
        elif x <= q2:
            return 1  # Moderate
        else:
            return 2  # Unhealthy

    df_master["AQI_class"] = df_master["AQI_raw"].apply(aqi_class_from_raw)

    return df_master


# -------------------------------------------------------------------
# 2. Sequence builder
# -------------------------------------------------------------------

def _build_sequences_classification(
    df: pd.DataFrame,
    feature_cols,
    label_col: str,
    seq_len: int,
):
    """
    Build LSTM-ready sequences (X, y) for classification.
    """
    X_list, y_list = [], []
    dates_list, regions_list = [], []

    for region in sorted(df["region"].unique()):
        df_r = df[df["region"] == region].sort_values("date").reset_index(drop=True)

        values = df_r[feature_cols].values
        labels = df_r[label_col].values
        dates = df_r["date"].values

        for t in range(seq_len, len(df_r)):
            X_list.append(values[t - seq_len : t, :])
            y_list.append(labels[t])
            dates_list.append(dates[t])
            regions_list.append(region)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=int)
    meta_dates = np.array(dates_list)
    meta_regions = np.array(regions_list)

    return X, y, meta_dates, meta_regions


# -------------------------------------------------------------------
# 3. Training entrypoint
# -------------------------------------------------------------------

def train_aqi_classifier(
    data_dir: Path = DATA_CSV_DIR,
    out_dir: Path | None = None,
    seq_len: int = 60,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    epochs: int = 30,
    batch_size: int = 32,
):
    """
    Train the LSTM AQI classifier and save:
      - model:     aqi_classifier.keras
      - scaler:    aqi_scaler.pkl
      - metadata:  aqi_metadata.json
    """
    if out_dir is None:
        out_dir = MODELS_DIR / "classification"
    ensure_dir(out_dir)

    print(f"[AQI] Loading raw data from: {data_dir}")
    dfs = _load_raw_aqi_data(data_dir)
    df_master = _build_master_df_classification(dfs)

    GAS_COLS = ["NO2", "CO", "CH4", "O3", "SO2"]
    EXTRA_COLS = ["LST_C", "gas_mean", "gas_sum", "sin_doy", "cos_doy"]
    FEATURE_COLS = GAS_COLS + EXTRA_COLS
    LABEL_COL = "AQI_class"

    X, y, dates, regions = _build_sequences_classification(
        df_master,
        FEATURE_COLS,
        LABEL_COL,
        seq_len,
    )

    order_idx = np.argsort(dates)
    X = X[order_idx]
    y = y[order_idx]
    dates = dates[order_idx]
    regions = regions[order_idx]

    N = X.shape[0]
    n_train = int(train_frac * N)
    n_val = int(val_frac * N)
    n_test = N - n_train - n_val

    X_train_raw = X[:n_train]
    y_train = y[:n_train]

    X_val_raw = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]

    X_test_raw = X[n_train + n_val :]
    y_test = y[n_train + n_val :]

    scaler_X = StandardScaler()
    n_features = X_train_raw.shape[2]

    X_train_flat = X_train_raw.reshape(-1, n_features)
    X_val_flat   = X_val_raw.reshape(-1, n_features)
    X_test_flat  = X_test_raw.reshape(-1, n_features)

    X_train = scaler_X.fit_transform(X_train_flat).reshape(X_train_raw.shape)
    X_val   = scaler_X.transform(X_val_flat).reshape(X_val_raw.shape)
    X_test  = scaler_X.transform(X_test_flat).reshape(X_test_raw.shape)

    n_timesteps = X_train.shape[1]
    num_classes = len(np.unique(y))

    model = Sequential(
        [
            Input(shape=(n_timesteps, n_features)),
            LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            Dropout(0.3),
            LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_path = out_dir / "aqi_classifier.keras"

    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        min_delta=1e-4,
        verbose=1,
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = dict(enumerate(class_weights))

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weight_dict,
        verbose=1,
    )

    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"[AQI] Test accuracy: {test_acc:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    with open(out_dir / "aqi_scaler.pkl", "wb") as f:
        pickle.dump(scaler_X, f)

    metadata = {
        "seq_len": seq_len,
        "n_features": n_features,
        "feature_names": FEATURE_COLS,
        "n_classes": int(num_classes),
        "class_names": {0: "Good", 1: "Moderate", 2: "Unhealthy"},
        "train_frac": train_frac,
        "val_frac": val_frac,
    }

    with open(out_dir / "aqi_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "test_accuracy": float(test_acc),
        "n_samples": int(N),
        "model_path": str(model_path),
        "scaler_path": str(out_dir / "aqi_scaler.pkl"),
    }


# -------------------------------------------------------------------
# 4. Inference helpers
# -------------------------------------------------------------------

def load_aqi_artifacts(
    models_dir: Path | None = None,
):
    """
    Load trained AQI classifier model + scaler + metadata from disk.
    """
    if models_dir is None:
        models_dir = MODELS_DIR / "classification"

    model = tf.keras.models.load_model(models_dir / "aqi_classifier.keras")
    with open(models_dir / "aqi_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / "aqi_metadata.json", "r") as f:
        metadata = json.load(f)

    return model, scaler, metadata


def classify_sequence(
    seq: np.ndarray,
    model,
    scaler: StandardScaler,
    metadata: dict,
):
    """
    Classify a single LSTM input sequence.
    """
    seq_len = metadata["seq_len"]
    n_features = metadata["n_features"]

    assert seq.shape == (seq_len, n_features), (
        f"Expected sequence shape {(seq_len, n_features)}, got {seq.shape}"
    )

    seq_scaled = scaler.transform(seq.reshape(-1, n_features)).reshape(
        1, seq_len, n_features
    )

    proba = model.predict(seq_scaled, verbose=0)[0]
    cls = int(np.argmax(proba))

    return cls, proba
