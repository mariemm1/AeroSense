// src/app/services/atmospheric-gases.service.ts

import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

import {
  LatestS5PResponse,
  LatestS3LSTResponse
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
}
