#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentinel-3 SLSTR L2 LST pipeline.

Similar structure to s5p_pipeline.py but only for LST.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import os

from pipeline_common import (
    AOI_BBOX,
    REGION_BBOXES,
    GRID_RES,
    ROOT_DIR,
    TIF_DIR,
    CSV_DIR,
    S3_LST,
    ensure_dir,
    wkt_polygon_from_bbox,
    bbox_to_geojson_polygon,
    aoi_key,
    yesterday_utc_range,
    auth_token,
    odata_search,
    download_product,
    is_zip_file,
    extract_zip_to_folder,
    tree_print,
    find_s3_lst_pair_explicit,
    find_s3_files_by_content,
    find_s3_flags_optional,
    open_nc_any,
    pick_first_var,
    apply_s3_lst_quality,
    grid_swath,
    write_geotiff,
    grid_stats,
    infer_scene_date,
    mongo_connect,
    mongo_ensure_indexes,
    mongo_upsert_many_s3,
)

import numpy as np
import pandas as pd


def process_s3_lst(
    token: str,
    bbox,
    region_name: str,
    start_date: str,
    end_date: str,
    top: int,
    out_root: Path,
    mongo_s3_coll=None,
):
    """
    Process Sentinel-3 LST for one region.
    """
    print(f"\n=== Region: {region_name} | S3 LST ===")
    footprint = wkt_polygon_from_bbox(bbox)
    product_ids = odata_search(
        token,
        S3_LST["collection"],
        S3_LST["token"],
        footprint,
        start_date,
        end_date,
        top=top,
    )
    if not product_ids:
        print("  ! No LST products found.")
        return []

    pkg_dir = out_root / f"{S3_LST['subdir']}_pkgs"
    work_dir = out_root / f"{S3_LST['subdir']}_nc"
    tif_dir = TIF_DIR / region_name / S3_LST["subdir"]
    ensure_dir(pkg_dir)
    ensure_dir(work_dir)
    ensure_dir(tif_dir)

    outputs = []
    mongo_docs = []
    _aoi_key = aoi_key(bbox, GRID_RES)
    _aoi_geojson = bbox_to_geojson_polygon(bbox)

    for pid in product_ids:
        pkg_path = pkg_dir / f"S3_LST_{region_name}_{pid}.zip"
        download_product(token, pid, pkg_path)

        if is_zip_file(pkg_path):
            out_folder = work_dir / f"S3_LST_{region_name}_{pid}"
            if not out_folder.exists() or not any(out_folder.rglob("*.nc")):
                extract_zip_to_folder(pkg_path, out_folder)

            tree_print(out_folder, max_files=150)

            lst_nc, geo_nc = find_s3_lst_pair_explicit(out_folder)
            if lst_nc is None or geo_nc is None:
                lst_nc, geo_nc = find_s3_files_by_content(
                    out_folder,
                    S3_LST["var_candidates"],
                    S3_LST["lat_candidates"],
                    S3_LST["lon_candidates"],
                )
            if lst_nc is None or geo_nc is None:
                print("  ! Could not locate LST_in.nc + geodetic_in.nc (or equivalents).")
                continue

            flags_nc = find_s3_flags_optional(out_folder)

            try:
                ds_lst = open_nc_any(lst_nc)
                ds_geo = open_nc_any(geo_nc)

                var = pick_first_var(ds_lst, S3_LST["var_candidates"])
                lat = pick_first_var(ds_geo, S3_LST["lat_candidates"])
                lon = pick_first_var(ds_geo, S3_LST["lon_candidates"])

                if flags_nc is not None:
                    try:
                        ds_flags = open_nc_any(flags_nc)
                        flag = pick_first_var(ds_flags, S3_LST["flag_candidates"])
                        good = (flag == 0) | np.isfinite(flag)
                        var = var.where(good)
                        ds_flags.close()
                    except Exception:
                        pass

                var = apply_s3_lst_quality(var, ds_lst)
                grid, transform = grid_swath(var, lat, lon, bbox, GRID_RES)

                if np.isfinite(grid).sum() == 0:
                    print("  ! No LST pixels fell inside your bbox for this pass.")
                    ds_lst.close()
                    ds_geo.close()
                    continue

                date_str = infer_scene_date(ds_lst, lst_nc.name)
                ds_lst.close()
                ds_geo.close()
            except Exception as e:
                print(f"  ! LST var selection error (package): {e}")
                continue

        else:
            try:
                ds = open_nc_any(pkg_path)
                var = pick_first_var(ds, S3_LST["var_candidates"])
                lat = pick_first_var(ds, S3_LST["lat_candidates"])
                lon = pick_first_var(ds, S3_LST["lon_candidates"])
                var = apply_s3_lst_quality(var, ds)
                grid, transform = grid_swath(var, lat, lon, bbox, GRID_RES)
                if np.isfinite(grid).sum() == 0:
                    print("  ! No LST pixels fell inside your bbox for this file.")
                    ds.close()
                    continue
                date_str = infer_scene_date(ds, pkg_path.name)
                ds.close()
            except Exception as e:
                print(f"  ! LST var selection error (single .nc): {e}")
                continue

        tif_path = tif_dir / f"LST_{region_name}_{date_str}.tif"
        write_geotiff(tif_path, grid, transform)
        stats = grid_stats(grid)
        outputs.append(
            {
                "region": region_name,
                "product": "LST",
                "date": date_str,
                "tif": str(tif_path),
                **stats,
            }
        )
        print(
            f"  ✓ LST {region_name} {date_str} "
            f"mean(K)={stats['mean']:.3f} (saved {tif_path.name})"
        )

        mongo_docs.append({
            "source": "S3",
            "region": region_name,
            "product": "LST",
            "date": date_str,
            "grid_res_deg": GRID_RES,
            "aoi_key": _aoi_key,
            "aoi": _aoi_geojson,
            "stats": stats,
            "files": {
                "tif": str(tif_path),
                "raw_pkg": str(pkg_path),
            },
            "params": {
                "product_token": S3_LST["token"],
                "bbox": bbox,
                "start": start_date,
                "end": end_date,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    if outputs:
        ensure_dir(CSV_DIR)
        csv_path = CSV_DIR / "s3_lst_daily.csv"
        df = pd.DataFrame(outputs)
        if csv_path.exists():
            old = pd.read_csv(csv_path)
            df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(
            subset=["region", "product", "date"], keep="last"
        ).sort_values(["region", "product", "date"])
        df.to_csv(csv_path, index=False)
        print(f"  → stats CSV updated: {csv_path}")
    else:
        print("  ! No LST scenes processed successfully; skipping stats CSV update.")

    if mongo_s3_coll is not None and mongo_docs:
        mongo_upsert_many_s3(mongo_s3_coll, mongo_docs)
        print(f"  → MongoDB upserted {len(mongo_docs)} LST doc(s).")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Sentinel-3 LST daily downloader/processor "
                    "(CDSE + optional MongoDB, per Tunisian region)."
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


    parser.add_argument("--minx", type=float, default=AOI_BBOX[0])
    parser.add_argument("--miny", type=float, default=AOI_BBOX[1])
    parser.add_argument("--maxx", type=float, default=AOI_BBOX[2])
    parser.add_argument("--maxy", type=float, default=AOI_BBOX[3])
    parser.add_argument(
        "--region-name",
        default="custom",
        help="Name used for Mongo/CSV when using manual bbox.",
    )

    parser.add_argument(
        "--regions",
        default="tunisia",
        help="Comma list of predefined regions (keys of REGION_BBOXES). "
             "If non-empty, overrides manual bbox.",
    )

    parser.add_argument("--start", default=None,
                        help="ISO start (default=yesterday 00:00Z)")
    parser.add_argument("--end", default=None,
                        help="ISO end (default=yesterday 23:59Z)")
    parser.add_argument("--top", type=int, default=1,
                        help="Latest N scenes per product")

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
        "--mongo-s3-col",
        default=os.getenv("MONGO_S3_COL", "s3_lst_daily"),
        help="Mongo collection for S3 LST daily stats",
    )

    args = parser.parse_args()

    if args.start and args.end:
        start_date, end_date, run_day = args.start, args.end, args.start[:10]
    else:
        start_date, end_date, run_day = yesterday_utc_range()

    print(f"Time: {start_date} → {end_date} (day {run_day})")
    print(f"TOP per product: {args.top}")

    ensure_dir(ROOT_DIR)
    ensure_dir(TIF_DIR)
    ensure_dir(CSV_DIR)

    mongo_db = mongo_client = None
    s3_coll = None
    if args.mongo_uri:
        mongo_db, mongo_client = mongo_connect(args.mongo_uri, args.mongo_db)
        if mongo_db is None:
            print("  ! Mongo disabled (connection failed or pymongo not installed).")
        else:
            s3_coll = mongo_db[args.mongo_s3_col]
            mongo_ensure_indexes(s3_coll, for_s5p=False)
            print(
                f"✓ Mongo connected: db='{args.mongo_db}', "
                f"col='{args.mongo_s3_col}'"
            )

    print("\nAuthenticating…")
    token = auth_token(args.user, args.password)
    print("✓ Auth OK")

    region_keys = [
        r.strip().lower() for r in args.regions.split(",") if r.strip()
    ]
    if not region_keys:
        region_keys = []

    if region_keys:
        for reg in region_keys:
            bbox = REGION_BBOXES.get(reg)
            if bbox is None:
                print(f"  ! Unknown region '{reg}' (skip).")
                continue
            print(f"\n=== Processing region '{reg}' with bbox {bbox} ===")
            process_s3_lst(
                token,
                bbox,
                region_name=reg,
                start_date=start_date,
                end_date=end_date,
                top=args.top,
                out_root=ROOT_DIR,
                mongo_s3_coll=s3_coll,
            )
    else:
        bbox = (args.minx, args.miny, args.maxx, args.maxy)
        region_name = args.region_name
        print(f"\n=== Processing manual region '{region_name}' with bbox {bbox} ===")
        process_s3_lst(
            token,
            bbox,
            region_name=region_name,
            start_date=start_date,
            end_date=end_date,
            top=args.top,
            out_root=ROOT_DIR,
            mongo_s3_coll=s3_coll,
        )

    if mongo_client:
        try:
            mongo_client.close()
        except Exception:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
