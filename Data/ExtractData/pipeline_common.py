#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common helpers for Sentinel-5P / Sentinel-3 pipelines.

Shared by:
  - s5p_pipeline.py
  - s3_pipeline.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import json
import zipfile
from typing import Optional, Tuple, List
import hashlib

import numpy as np
import pandas as pd
import requests
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from dotenv import load_dotenv

# --------------------------
# Load .env from backend
# --------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # AEROSENSE/
BACKEND_ENV = PROJECT_ROOT / "Backend" / ".env"

if BACKEND_ENV.exists():
    load_dotenv(BACKEND_ENV)
# local .env near this script (optional override)
load_dotenv()

# ---- MongoDB (optional) ----
try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

# -----------------------
# URLs & Global Defaults
# -----------------------

COPERNICUS_AUTH_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL_TPL = (
    "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
)

# Whole Tunisia (approx) – original default
AOI_BBOX = (7.5, 30.0, 12.0, 37.5)

# Approximate bboxes for some Tunisian governorates
# → You can refine these later with official shapefiles.
REGION_BBOXES = {
    "tunisia": AOI_BBOX,
    "ariana": (10.0, 36.7, 10.35, 37.0),
    "manouba": (9.8, 36.6, 10.2, 37.1),
    "siliana": (8.9, 35.8, 10.0, 36.7),
    "tozeur": (7.5, 33.4, 9.0, 34.2),
    # add more here...
}

# Output folders (relative to where scripts are run)
ROOT_DIR = Path("./data_copernicus")
TIF_DIR = ROOT_DIR / "tiffs"
CSV_DIR = ROOT_DIR / "stats"

GRID_RES = 0.05     # degrees
QA_THRESH = 0.75    # S5P qa_value threshold

# Sentinel-5P products: GAS -> (token in product Name, subfolder, variable candidates)
S5P_PRODUCTS = {
    "NO2": ("L2__NO2___", "no2", [
        "nitrogendioxide_tropospheric_column",
        "tropospheric_NO2_column_number_density",
    ]),
    "CO": ("L2__CO____", "co", [
        "carbonmonoxide_total_column",
        "carbonmonoxide_column_number_density",
    ]),
    "CH4": ("L2__CH4___", "ch4", [
        "methane_mixing_ratio_bias_corrected",
        "methane_mixing_ratio",
    ]),
    "O3": ("L2__O3____", "o3", [
        "ozone_total_vertical_column",
        "ozone_total_vertical_column_tropospheric_corrected",
    ]),
    "SO2": ("L2__SO2___", "so2", [
        "sulfurdioxide_total_vertical_column",
        "sulfurdioxide_column_number_density",
    ]),
}

# Sentinel-3 SLSTR L2 LST description
S3_LST = {
    "collection": "SENTINEL-3",
    "token": "SL_2_LST___",
    "subdir": "lst",
    "var_candidates": ["LST", "lst", "surface_temperature", "sea_surface_temperature"],
    "lat_candidates": ["latitude_in", "latitude", "lat"],
    "lon_candidates": ["longitude_in", "longitude", "lon"],
    "flag_candidates": ["confidence_in", "quality_in", "L2_flags", "flags"],
}

# -------------
# Small Utils
# -------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def wkt_polygon_from_bbox(bbox):
    minx, miny, maxx, maxy = bbox
    return (
        f"POLYGON(({minx} {miny}, {minx} {maxy}, {maxx} {maxy}, "
        f"{maxx} {miny}, {minx} {miny}))"
    )


def bbox_to_geojson_polygon(bbox):
    minx, miny, maxx, maxy = bbox
    coords = [[
        [minx, miny],
        [minx, maxy],
        [maxx, maxy],
        [maxx, miny],
        [minx, miny],
    ]]
    return {"type": "Polygon", "coordinates": coords}


def aoi_key(bbox, res):
    raw = json.dumps({"bbox": bbox, "res": res}, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def yesterday_utc_range():
    today = datetime.now(timezone.utc).date()
    y = today - timedelta(days=1)
    start = (
        datetime(y.year, y.month, y.day, 0, 0, 0, tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    end = (
        datetime(y.year, y.month, y.day, 23, 59, 59, tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return start, end, y.isoformat()

# -------------
# Auth & OData
# -------------

def auth_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
        "scope": "openid",
    }
    r = requests.post(COPERNICUS_AUTH_URL, data=data, timeout=60)
    if r.status_code != 200:
        try:
            print("Auth error detail:", r.json())
        except Exception:
            print("Auth error text:", r.text[:400])
        r.raise_for_status()
    return r.json()["access_token"]


def odata_search(token, collection_name, product_token, footprint_wkt,
                 start_date, end_date, top=1):
    headers = {"Authorization": f"Bearer {token}"}
    filter_q = (
        f"Collection/Name eq '{collection_name}' and "
        f"contains(Name, '{product_token}') and "
        f"OData.CSC.Intersects(Footprint, geography'SRID=4326;{footprint_wkt}') and "
        f"ContentDate/Start ge {start_date} and "
        f"ContentDate/End le {end_date}"
    )
    params = {"$filter": filter_q, "$top": top, "$format": "json"}
    r = requests.get(CATALOG_URL, headers=headers, params=params, timeout=120)
    r.raise_for_status()
    items = r.json().get("value", [])
    return [item["Id"] for item in items]


def download_product(token, product_id, out_path: Path):
    headers = {"Authorization": f"Bearer {token}"}
    url = DOWNLOAD_URL_TPL.format(product_id=product_id)
    ensure_dir(out_path.parent)
    if out_path.exists():
        print(f"  • exists: {out_path.name}")
        return out_path

    with requests.get(url, headers=headers, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    print(f"  ✓ downloaded: {out_path.name}")
    return out_path

# -------------------------
# NetCDF helpers
# -------------------------

def open_nc_any(nc_path: Path, group: str | None = None):
    last_err = None
    for engine in ("h5netcdf", "netcdf4", None):
        try:
            if group is None:
                return xr.open_dataset(nc_path, engine=engine)
            return xr.open_dataset(nc_path, engine=engine, group=group)
        except Exception as e:
            last_err = e
    raise last_err


def open_s5p_groups(nc_path: Path):
    ds_prod = open_nc_any(nc_path, group="PRODUCT")
    try:
        lat = ds_prod["latitude"]
        lon = ds_prod["longitude"]
    except KeyError:
        ds_geo = open_nc_any(nc_path, group="GEOLOCATION")
        lat = ds_geo["latitude"]
        lon = ds_geo["longitude"]
        ds_prod._geo = ds_geo
    return ds_prod, lat, lon

# ---------- ZIP helpers (S3) ----------

def is_zip_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"PK"
    except Exception:
        return False


def extract_zip_to_folder(zip_path: Path, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def tree_print(root: Path, max_files: int = 200):
    print("  • Package content (first items):")
    n = 0
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        print(f"    - {rel}")
        n += 1
        if n >= max_files:
            print("    ...")
            break


def has_var(ds: xr.Dataset, names: List[str]) -> Optional[str]:
    lower = {k.lower(): k for k in list(ds.variables)}
    for candidate in names:
        key = candidate.lower()
        if key in lower:
            return lower[key]
        for lv, orig in lower.items():
            if key in lv:
                return orig
    return None


def open_nc_any_silent(path: Path) -> Optional[xr.Dataset]:
    for engine in ("h5netcdf", "netcdf4", None):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception:
            pass
    return None


def find_s3_files_by_content(
    folder: Path,
    lst_var_candidates: List[str],
    lat_candidates: List[str],
    lon_candidates: List[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    lst_nc = None
    geo_nc = None
    for p in folder.rglob("*.nc"):
        ds = open_nc_any_silent(p)
        if ds is None:
            continue
        try:
            if lst_nc is None:
                v = has_var(ds, lst_var_candidates)
                if v:
                    lst_nc = p
            if geo_nc is None:
                lat_ok = has_var(ds, lat_candidates)
                lon_ok = has_var(ds, lon_candidates)
                if not (lat_ok and lon_ok):
                    lat_ok = lon_ok = False
                    for _, da in ds.variables.items():
                        std = str(da.attrs.get("standard_name", "")).lower()
                        if std == "latitude":
                            lat_ok = True
                        if std == "longitude":
                            lon_ok = True
                    if lat_ok and lon_ok:
                        geo_nc = p
                else:
                    geo_nc = p
        finally:
            ds.close()
        if lst_nc is not None and geo_nc is not None:
            break
    return lst_nc, geo_nc


def find_s3_lst_pair_explicit(folder: Path):
    lst_nc = None
    geo_nc = None
    for p in folder.rglob("*.nc"):
        n = p.name.lower()
        if n == "lst_in.nc":
            lst_nc = p
        elif n == "geodetic_in.nc":
            geo_nc = p
    return lst_nc, geo_nc


def find_s3_flags_optional(folder: Path):
    for p in folder.rglob("*.nc"):
        n = p.name.lower()
        if n == "flags_in.nc" or "flags" in n:
            return p
    return None

# -------------------------
# Mongo helpers
# -------------------------

def mongo_safe_create_index(coll, keys, name=None, **opts):
    try:
        return coll.create_index(keys, name=name, **opts)
    except Exception as e:
        if "Index already exists with a different name" in str(e):
            return None
        raise


def mongo_ensure_indexes(coll, for_s5p: bool):
    if coll is None:
        return
    if for_s5p:
        mongo_safe_create_index(
            coll,
            [("source", 1), ("gas", 1), ("date", 1),
             ("aoi_key", 1), ("grid_res_deg", 1)],
            unique=True,
            name="uniq_s5p_daily",
        )
    else:
        mongo_safe_create_index(
            coll,
            [("source", 1), ("date", 1),
             ("aoi_key", 1), ("grid_res_deg", 1)],
            unique=True,
            name="uniq_s3_daily",
        )
    mongo_safe_create_index(coll, [("date", 1)], name="by_date")


def mongo_connect(uri: str | None, db_name: str | None):
    if not uri or MongoClient is None:
        return None, None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _ = client.server_info()
        db = client[db_name]
        return db, client
    except Exception as e:
        print(f"  ! Mongo connect error: {e}")
        return None, None


def mongo_upsert_many_s5p(coll, docs: list[dict]):
    if coll is None or not docs:
        return
    for d in docs:
        filt = {
            "source": d["source"],
            "gas": d["gas"],
            "date": d["date"],
            "aoi_key": d["aoi_key"],
            "grid_res_deg": d["grid_res_deg"],
        }
        try:
            coll.update_one(filt, {"$set": d}, upsert=True)
        except Exception as e:
            print(f"  ! Mongo upsert S5P error: {e}")


def mongo_upsert_many_s3(coll, docs: list[dict]):
    if coll is None or not docs:
        return
    for d in docs:
        filt = {
            "source": d["source"],
            "date": d["date"],
            "aoi_key": d["aoi_key"],
            "grid_res_deg": d["grid_res_deg"],
        }
        try:
            coll.update_one(filt, {"$set": d}, upsert=True)
        except Exception as e:
            print(f"  ! Mongo upsert S3 error: {e}")

# -------------------------
# Variable picking & QA
# -------------------------

def pick_first_var(ds: xr.Dataset, candidates):
    for name in candidates:
        if name in ds.variables:
            return ds[name]
    for v in ds.data_vars:
        for name in candidates:
            if name.lower() in v.lower():
                return ds[v]
    raise KeyError(
        f"None of {candidates} found in dataset vars: {list(ds.data_vars)[:12]}..."
    )


def apply_s5p_quality(var: xr.DataArray, ds: xr.Dataset, qa_thresh=QA_THRESH):
    qa = ds.get("qa_value")
    if qa is not None:
        return var.where(qa >= qa_thresh)
    return var


def apply_s3_lst_quality(var: xr.DataArray, ds: xr.Dataset):
    return var.where((var >= 250.0) & (var <= 350.0))

# -------------------------
# Gridding & GeoTIFF I/O
# -------------------------

def grid_swath(var: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray,
               bbox, res=GRID_RES):
    minx, miny, maxx, maxy = bbox

    v = var.values
    la = lat.values
    lo = lon.values

    mask = np.isfinite(v)
    v, la, lo = v[mask], la[mask], lo[mask]

    inb = (lo >= minx) & (lo <= maxx) & (la >= miny) & (la <= maxy)
    v, la, lo = v[inb], la[inb], lo[inb]

    x_bins = np.arange(minx, maxx + res, res)
    y_bins = np.arange(miny, maxy + res, res)

    if v.size == 0:
        grid = np.full(
            (len(y_bins) - 1, len(x_bins) - 1),
            np.nan,
            dtype="float32",
        )
        transform = from_origin(x_bins[0], y_bins[-1], res, res)
        return grid, transform

    H, _, _ = np.histogram2d(lo, la, bins=[x_bins, y_bins], weights=v)
    C, _, _ = np.histogram2d(lo, la, bins=[x_bins, y_bins])
    grid = H / np.maximum(C, 1)

    transform = from_origin(x_bins[0], y_bins[-1], res, res)
    return grid.T.astype("float32"), transform


def write_geotiff(path: Path, grid: np.ndarray, transform, crs="EPSG:4326"):
    ensure_dir(path.parent)
    profile = {
        "driver": "GTiff",
        "height": grid.shape[0],
        "width": grid.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(grid, 1)


def grid_stats(grid: np.ndarray) -> dict:
    return {
        "mean": float(np.nanmean(grid)),
        "median": float(np.nanmedian(grid)),
        "min": float(np.nanmin(grid)),
        "max": float(np.nanmax(grid)),
        "valid_pixels": int(np.isfinite(grid).sum()),
    }


def infer_scene_date(ds: xr.Dataset, fallback_name: str) -> str:
    for k in ["time", "start_time", "time_utc", "TIME", "datetime"]:
        if k in ds.coords or k in ds.variables:
            try:
                val = ds[k].values
                s = str(val if np.ndim(val) == 0 else val[0])
                return s[:10]
            except Exception:
                pass
    digits = "".join(c for c in fallback_name if c.isdigit())
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return datetime.now(timezone.utc).date().isoformat()
