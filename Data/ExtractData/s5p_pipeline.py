#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentinel-5P pipeline (NO2, CO, CH4, O3, SO2).

Processes:
  - a chosen AOI (either manual bbox or predefined region(s) in Tunisia)
  - selected gases
  - writes GeoTIFFs + CSV
  - upserts daily stats into MongoDB (same DB as Django backend)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os

from pipeline_common import (
    # constants
    AOI_BBOX,
    REGION_BBOXES,
    GRID_RES,
    QA_THRESH,
    ROOT_DIR,
    TIF_DIR,
    CSV_DIR,
    S5P_PRODUCTS,
    # utils
    ensure_dir,
    wkt_polygon_from_bbox,
    bbox_to_geojson_polygon,
    aoi_key,
    yesterday_utc_range,
    auth_token,
    odata_search,
    download_product,
    open_s5p_groups,
    pick_first_var,
    apply_s5p_quality,
    grid_swath,
    write_geotiff,
    grid_stats,
    infer_scene_date,
    mongo_connect,
    mongo_ensure_indexes,
    mongo_upsert_many_s5p,
)

import pandas as pd


def process_s5p_gas(
    token: str,
    gas_key: str,
    product_token: str,
    var_candidates,
    bbox,
    region_name: str,
    start_date: str,
    end_date: str,
    top: int,
    out_root: Path,
    qa_thresh: float,
    mongo_s5p_coll=None,
):
    """
    For a given gas in a given region:
      - search latest N S5P L2 scenes intersecting AOI
      - download .nc
      - grid & save GeoTIFF
      - write CSV stats
      - upsert daily stats to Mongo (if enabled), tagged with 'region'
    """
    print(f"\n=== Region: {region_name} | S5P {gas_key} ===")
    footprint = wkt_polygon_from_bbox(bbox)
    product_ids = odata_search(
        token,
        "SENTINEL-5P",
        product_token,
        footprint,
        start_date,
        end_date,
        top=top,
    )
    if not product_ids:
        print("  ! No products found.")
        return []

    nc_dir = out_root / f"{gas_key.lower()}_nc"
    tif_dir = TIF_DIR / region_name / gas_key.lower()
    ensure_dir(nc_dir)
    ensure_dir(tif_dir)

    outputs = []
    mongo_docs = []
    _aoi_key = aoi_key(bbox, GRID_RES)
    _aoi_geojson = bbox_to_geojson_polygon(bbox)

    for pid in product_ids:
        nc_path = nc_dir / f"S5P_{gas_key}_{region_name}_{pid}.nc"
        result_path = download_product(token, pid, nc_path)
        if result_path is None:
            # download failed, skip this product
            continue

        try:
            ds, lat, lon = open_s5p_groups(result_path)
            var = pick_first_var(ds, var_candidates)
            var = apply_s5p_quality(var, ds, qa_thresh)
        except Exception as e:
            print(f"  ! Var selection error: {e}")
            try:
                print("    • PRODUCT vars:", list(ds.data_vars)[:15])
            except Exception:
                pass
            try:
                ds.close()
            except Exception:
                pass
            continue

        grid, transform = grid_swath(var, lat, lon, bbox, GRID_RES)
        date_str = infer_scene_date(ds, result_path.name)
        ds.close()

        tif_path = tif_dir / f"{gas_key}_{region_name}_{date_str}.tif"
        write_geotiff(tif_path, grid, transform)
        stats = grid_stats(grid)
        outputs.append(
            {
                "region": region_name,
                "gas": gas_key,
                "date": date_str,
                "tif": str(tif_path),
                **stats,
            }
        )
        print(
            f"  ✓ {gas_key} {region_name} {date_str} "
            f"mean={stats['mean']:.6g} (saved {tif_path.name})"
        )

        mongo_docs.append(
            {
                "source": "S5P",
                "region": region_name,
                "gas": gas_key,
                "date": date_str,
                "grid_res_deg": GRID_RES,
                "aoi_key": _aoi_key,
                "aoi": _aoi_geojson,
                "stats": stats,
                "files": {
                    "tif": str(tif_path),
                    "raw_nc": str(result_path),
                },
                "params": {
                    "qa_threshold": qa_thresh,
                    "product_token": product_token,
                    "bbox": bbox,
                    "start": start_date,
                    "end": end_date,
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    if outputs:
        ensure_dir(CSV_DIR)
        csv_path = CSV_DIR / f"s5p_{gas_key.lower()}_daily.csv"
        df = pd.DataFrame(outputs)
        if csv_path.exists():
            old = pd.read_csv(csv_path)
            df = pd.concat([old, df], ignore_index=True)
        df = (
            df.drop_duplicates(subset=["region", "gas", "date"], keep="last")
            .sort_values(["region", "gas", "date"])
        )
        df.to_csv(csv_path, index=False)
        print(f"  → stats CSV updated: {csv_path}")
    else:
        print("  ! No scenes processed successfully; skipping stats CSV update.")

    if mongo_s5p_coll is not None and mongo_docs:
        mongo_upsert_many_s5p(mongo_s5p_coll, mongo_docs)
        print(f"  → MongoDB upserted {len(mongo_docs)} S5P doc(s).")

    return outputs


def parse_date_only(s: str):
    """
    Helper: get a date (YYYY-MM-DD) from different ISO-like strings.
    Accepts 'YYYY-MM-DD' or full 'YYYY-MM-DDTHH:MM:SSZ'.
    """
    ds = s[:10]
    return datetime.strptime(ds, "%Y-%m-%d").date()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sentinel-5P gases daily downloader/processor "
            "(CDSE + optional MongoDB, per Tunisian region)."
        )
    )
    parser.add_argument(
        "--user",
        default=os.getenv("CDSE_USER"),
        help="CDSE username or email (default from CDSE_USER)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("CDSE_PASSWORD"),
        help="CDSE password (default from CDSE_PASSWORD)",
    )

    # manual bbox (used if --regions is empty)
    parser.add_argument("--minx", type=float, default=AOI_BBOX[0])
    parser.add_argument("--miny", type=float, default=AOI_BBOX[1])
    parser.add_argument("--maxx", type=float, default=AOI_BBOX[2])
    parser.add_argument("--maxy", type=float, default=AOI_BBOX[3])
    parser.add_argument(
        "--region-name",
        default="custom",
        help="Name used for Mongo/CSV when using manual bbox.",
    )

    # list of predefined regions, e.g. "tunisia,ariana,tozeur" or "all"
    parser.add_argument(
        "--regions",
        default="tunisia",
        help=(
            "Comma list of predefined regions (keys of REGION_BBOXES) "
            "or 'all' for every predefined region. "
            "If non-empty, overrides manual bbox."
        ),
    )

    parser.add_argument(
        "--start",
        default=None,
        help="ISO start (default=yesterday 00:00Z)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="ISO end (default=yesterday 23:59Z)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1,
        help="Latest N scenes per query/filter",
    )
    parser.add_argument(
        "--gases",
        default="NO2,CO,CH4,O3,SO2",
        help="Comma list of S5P gases to process",
    )
    parser.add_argument(
        "--qa",
        type=float,
        default=QA_THRESH,
        help="QA threshold for S5P (default 0.75)",
    )

    # DAILY mode
    parser.add_argument(
        "--daily",
        action="store_true",
        help=(
            "If set, loop day-by-day between start and end, "
            "using a 00:00Z–23:59Z window for each day. "
            "Typically use --top 1 in this mode."
        ),
    )

    # Mongo options
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGO_URI"),
        help="MongoDB URI (omit to disable Mongo)",
    )
    parser.add_argument(
        "--mongo-db",
        default=os.getenv("MONGO_DB", "copernicus"),
        help="Mongo database name",
    )
    parser.add_argument(
        "--mongo-s5p-col",
        default=os.getenv("MONGO_S5P_COL", "s5p_daily"),
        help="Mongo collection for S5P daily stats",
    )

    args = parser.parse_args()

    # time window (base strings)
    if args.start and args.end:
        start_iso, end_iso, run_day = args.start, args.end, args.start[:10]
    else:
        start_iso, end_iso, run_day = yesterday_utc_range()

    print(f"Time: {start_iso} → {end_iso} (day {run_day})")
    print(f"TOP per product: {args.top}")
    if args.daily:
        print("Mode: DAILY (loop per day)")
    else:
        print("Mode: RANGE (single query per gas/region)")

    ensure_dir(ROOT_DIR)
    ensure_dir(TIF_DIR)
    ensure_dir(CSV_DIR)

    # Mongo
    mongo_db = mongo_client = None
    s5p_coll = None
    if args.mongo_uri:
        mongo_db, mongo_client = mongo_connect(args.mongo_uri, args.mongo_db)
        if mongo_db is None:
            print("  ! Mongo disabled (connection failed or pymongo not installed).")
        else:
            s5p_coll = mongo_db[args.mongo_s5p_col]
            mongo_ensure_indexes(s5p_coll, for_s5p=True)
            print(
                f"✓ Mongo connected: db='{args.mongo_db}', "
                f"col='{args.mongo_s5p_col}'"
            )

    # Auth
    print("\nAuthenticating…")
    token = auth_token(args.user, args.password)
    print("✓ Auth OK")

    # resolve regions
    region_keys = [
        r.strip().lower() for r in args.regions.split(",") if r.strip()
    ]

    # Special keyword: 'all' → all predefined regions in REGION_BBOXES
    if "all" in region_keys:
        region_keys = list(REGION_BBOXES.keys())

    if not region_keys:
        # fallback to manual bbox
        region_keys = []

    gases = [g.strip().upper() for g in args.gases.split(",") if g.strip()]

    # DAILY MODE: loop over days
    if args.daily and args.start and args.end:
        start_date = parse_date_only(start_iso)
        end_date = parse_date_only(end_iso)

        if start_date > end_date:
            raise ValueError("start date must be <= end date in daily mode.")

        current = start_date
        while current <= end_date:
            day_start = f"{current.isoformat()}T00:00:00Z"
            day_end = f"{current.isoformat()}T23:59:59Z"
            print(
                f"\n--- DAILY window {day_start} → {day_end} "
                f"({current.isoformat()}) ---"
            )

            if region_keys:
                # predefined regions
                for reg in region_keys:
                    bbox = REGION_BBOXES.get(reg)
                    if bbox is None:
                        print(f"  ! Unknown region '{reg}' (skip).")
                        continue
                    print(f"\n=== Processing region '{reg}' with bbox {bbox} ===")
                    for g in gases:
                        if g not in S5P_PRODUCTS:
                            print(f"  ! Unknown gas '{g}' (skip).")
                            continue
                        p_token, _subdir, var_candidates = S5P_PRODUCTS[g]
                        process_s5p_gas(
                            token,
                            g,
                            p_token,
                            var_candidates,
                            bbox,
                            region_name=reg,
                            start_date=day_start,
                            end_date=day_end,
                            top=args.top,
                            out_root=ROOT_DIR,
                            qa_thresh=args.qa,
                            mongo_s5p_coll=s5p_coll,
                        )
            else:
                # manual bbox
                bbox = (args.minx, args.miny, args.maxx, args.maxy)
                region_name = args.region_name
                print(
                    f"\n=== Processing manual region '{region_name}' "
                    f"with bbox {bbox} ==="
                )
                for g in gases:
                    if g not in S5P_PRODUCTS:
                        print(f"  ! Unknown gas '{g}' (skip).")
                        continue
                    p_token, _subdir, var_candidates = S5P_PRODUCTS[g]
                    process_s5p_gas(
                        token,
                        g,
                        p_token,
                        var_candidates,
                        bbox,
                        region_name=region_name,
                        start_date=day_start,
                        end_date=day_end,
                        top=args.top,
                        out_root=ROOT_DIR,
                        qa_thresh=args.qa,
                        mongo_s5p_coll=s5p_coll,
                    )

            current += timedelta(days=1)

    else:
        # ORIGINAL RANGE MODE (single query per gas/region)
        if region_keys:
            # use predefined Tunisian regions
            for reg in region_keys:
                bbox = REGION_BBOXES.get(reg)
                if bbox is None:
                    print(f"  ! Unknown region '{reg}' (skip).")
                    continue
                print(f"\n=== Processing region '{reg}' with bbox {bbox} ===")
                for g in gases:
                    if g not in S5P_PRODUCTS:
                        print(f"  ! Unknown gas '{g}' (skip).")
                        continue
                    p_token, _subdir, var_candidates = S5P_PRODUCTS[g]
                    process_s5p_gas(
                        token,
                        g,
                        p_token,
                        var_candidates,
                        bbox,
                        region_name=reg,
                        start_date=start_iso,
                        end_date=end_iso,
                        top=args.top,
                        out_root=ROOT_DIR,
                        qa_thresh=args.qa,
                        mongo_s5p_coll=s5p_coll,
                    )
        else:
            # manual bbox
            bbox = (args.minx, args.miny, args.maxx, args.maxy)
            region_name = args.region_name
            print(
                f"\n=== Processing manual region '{region_name}' "
                f"with bbox {bbox} ==="
            )
            for g in gases:
                if g not in S5P_PRODUCTS:
                    print(f"  ! Unknown gas '{g}' (skip).")
                    continue
                p_token, _subdir, var_candidates = S5P_PRODUCTS[g]
                process_s5p_gas(
                    token,
                    g,
                    p_token,
                    var_candidates,
                    bbox,
                    region_name=region_name,
                    start_date=start_iso,
                    end_date=end_iso,
                    top=args.top,
                    out_root=ROOT_DIR,
                    qa_thresh=args.qa,
                    mongo_s5p_coll=s5p_coll,
                )

    if mongo_client:
        try:
            mongo_client.close()
        except Exception:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
