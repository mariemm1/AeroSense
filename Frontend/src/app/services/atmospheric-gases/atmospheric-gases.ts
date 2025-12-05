// src/app/services/atmospheric-gases.service.ts

import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

import {
  LatestS5PResponse,
  LatestS3LSTResponse,
  ForecastResponse,
  AQIResponse
} from '../../models/atmospheric-gases.model';

@Injectable({
  providedIn: 'root'
})
export class AtmosphericGasesService {
  private readonly http = inject(HttpClient);

  // TODO: move to environment.ts when ready
  private readonly baseUrl = 'http://127.0.0.1:8000';

  /**
   * GET /api/s5p/latest/?region=&gas=&top=
   */
  getLatestS5P(
    region: string,
    gases?: string[],
    top: number = 7
  ): Observable<LatestS5PResponse> {
    let params = new HttpParams()
      .set('region', region.toLowerCase())
      .set('top', String(top));

    if (gases && gases.length) {
      params = params.set('gas', gases.join(','));
    }

    return this.http.get<LatestS5PResponse>(
      `${this.baseUrl}/api/s5p/latest/`,
      { params }
    );
  }

  /**
   * GET /api/s3/lst/latest/?region=&top=
   */
  getLatestS3LST(
    region: string,
    top: number = 7
  ): Observable<LatestS3LSTResponse> {
    const params = new HttpParams()
      .set('region', region.toLowerCase())
      .set('top', String(top));

    return this.http.get<LatestS3LSTResponse>(
      `${this.baseUrl}/api/s3/lst/latest/`,
      { params }
    );
  }

  /**
   * GET /api/forecast/?region=&last_date=
   * Backend should:
   *  - use latest data for this region
   *  - predict "tomorrow"
   *  - optionally return class_name / probabilities for forecasted air
   */
  getForecast(region: string, lastDate?: string): Observable<ForecastResponse> {
    let params = new HttpParams().set('region', region.toLowerCase());
    if (lastDate) {
      params = params.set('last_date', lastDate);
    }
    return this.http.get<ForecastResponse>(
      `${this.baseUrl}/api/forecast/`,
      { params }
    );
  }

  /**
   * GET /api/aqi/?region=
   * Backend should classify latest gases + LST for this region.
   */
  getAQI(region: string): Observable<AQIResponse> {
    const params = new HttpParams().set('region', region.toLowerCase());
    return this.http.get<AQIResponse>(
      `${this.baseUrl}/api/aqi/`,
      { params }
    );
  }
}
