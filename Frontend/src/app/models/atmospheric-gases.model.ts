// src/app/models/atmospheric-gases.model.ts

// -----------------------------------------------------------------------------
// Shared value / geometry helpers
// -----------------------------------------------------------------------------

export interface GridStats {
  mean: number;
  median: number;
  min: number;
  max: number;
  valid_pixels: number;
}

export interface GeoJsonPolygon {
  type: 'Polygon';
  coordinates: number[][][]; // [ [ [lon, lat], ... ] ]
}

// -----------------------------------------------------------------------------
// Sentinel-5P (atmospheric gases)
// -----------------------------------------------------------------------------

export interface S5PFiles {
  tif: string;
  raw_nc: string;
}

export interface S5PParams {
  qa_threshold?: number;
  product_token?: string;
  bbox?: number[]; // [minx, miny, maxx, maxy]
  start?: string;  // ISO8601
  end?: string;    // ISO8601
  [key: string]: any;
}

export interface S5PDocument {
  _id?: string;
  source: 'S5P';
  region: string;
  gas: string; // "NO2", "CO", "O3", "SO2", "CH4", ...
  date: string; // "YYYY-MM-DD"
  grid_res_deg: number;
  aoi_key: string;
  aoi: GeoJsonPolygon | any;
  stats: GridStats;
  files: S5PFiles;
  params: S5PParams;
  created_at: string;
}

export interface LatestS5PResponse {
  region: string;
  gases: string[];
  top: number;
  results: { [gas: string]: S5PDocument[] };
}

// -----------------------------------------------------------------------------
// Sentinel-3 LST (land surface temperature)
// -----------------------------------------------------------------------------

export interface S3Files {
  tif: string;
  raw_pkg: string;
}

export interface S3Params {
  product_token?: string;
  bbox?: number[];
  start?: string;
  end?: string;
  [key: string]: any;
}

export interface S3LSTDocument {
  _id?: string;
  source: 'S3';
  region: string;
  product: 'LST';
  date: string; // "YYYY-MM-DD"
  grid_res_deg: number;
  aoi_key: string;
  aoi: GeoJsonPolygon | any;
  stats: GridStats;
  files: S3Files;
  params: S3Params;
  created_at: string;
}

export interface LatestS3LSTResponse {
  region: string;
  top: number;
  results: S3LSTDocument[];
}
