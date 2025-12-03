import { ChangeDetectionStrategy, Component, OnInit} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { NgApexchartsModule, ApexOptions } from 'ng-apexcharts';
import { forkJoin } from 'rxjs';
import { SharedModule } from '../theme/shared/shared.module';
import { LatestS3LSTResponse,
  LatestS5PResponse
} from '../models/atmospheric-gases.model';
import { AtmosphericGasesService } from '../services/atmospheric-gases/atmospheric-gases';

interface AQICard {
  pollutant: string;
  key: string;
  value: number | null;
  unit: string;
  level: string;
  bgClass: string;
  message?: string;
}

interface SummaryCard {
  title: string;
  value: string;
  label: string;
  trend: 'up' | 'down' | 'neutral';
}

interface GasStats {
  mean: number;
  min: number;
  max: number;
}

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, SharedModule, NgApexchartsModule],
  templateUrl: './dashboard.html',
  styleUrls: ['./dashboard.css'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class Dashboard implements OnInit {
  // user & menu
  userName = 'Admin';
  userMenuOpen = false;

  toggleUserMenu(): void {
    this.userMenuOpen = !this.userMenuOpen;
  }
  closeUserMenu(): void {
    this.userMenuOpen = false;
  }

  // regions
  regions = [
    { id: 'tunisia', label: 'Tunisia (country)' },
    { id: 'tunis', label: 'Tunis' },
    { id: 'sfax', label: 'Sfax' },
    { id: 'tozeur', label: 'Tozeur' }
  ];
  selectedRegion = 'tunisia';

  isLoading = false;
  errorMessage: string | null = null;

  latestS5p: LatestS5PResponse | null = null;
  latestS3: LatestS3LSTResponse | null = null;

  // gas metadata (units / labels)
  readonly gasMeta: Record<string, { label: string; unit: string }> = {
    NO2: { label: 'NO₂', unit: 'µmol/m²' },
    O3: { label: 'O₃', unit: 'µg/m³' },
    SO2: { label: 'SO₂', unit: 'µg/m³' },
    CO: { label: 'CO', unit: 'mg/m³' },
    CH4: { label: 'CH₄', unit: 'ppb' },
    LST: { label: 'LST', unit: '°C' }
  };

  // AQI mini-cards (NO2, O3, SO2, CO, CH4, LST)
  aqiCards: AQICard[] = [
    {
      pollutant: 'NO₂',
      key: 'NO2',
      value: null,
      unit: this.gasMeta['NO2'].unit,
      level: 'No data',
      bgClass: 'aqi-bg-no2'
    },
    {
      pollutant: 'O₃',
      key: 'O3',
      value: null,
      unit: this.gasMeta['O3'].unit,
      level: 'No data',
      bgClass: 'aqi-bg-o3'
    },
    {
      pollutant: 'SO₂',
      key: 'SO2',
      value: null,
      unit: this.gasMeta['SO2'].unit,
      level: 'No data',
      bgClass: 'aqi-bg-so2'
    },
    {
      pollutant: 'CO',
      key: 'CO',
      value: null,
      unit: this.gasMeta['CO'].unit,
      level: 'No data',
      bgClass: 'aqi-bg-co'
    },
    {
      pollutant: 'CH₄',
      key: 'CH4',
      value: null,
      unit: this.gasMeta['CH4'].unit,
      level: 'No data',
      bgClass: 'aqi-bg-ch4'
    },
    {
      pollutant: 'LST',
      key: 'LST',
      value: null,
      unit: this.gasMeta['LST'].unit,
      level: 'No data',
      bgClass: 'aqi-bg-lst'
    }
  ];

  // summary KPI cards (still static/demo)
  summaryCards: SummaryCard[] = [
    { title: 'Daily AQI', value: '82', label: 'Current day index', trend: 'up' },
    { title: 'Weekly Avg AQI', value: '76', label: 'Last 7 days', trend: 'neutral' },
    { title: 'Monthly Avg AQI', value: '69', label: 'Last 30 days', trend: 'down' },
    { title: 'Coverage', value: '235 / 26', label: 'Grid cells / regions', trend: 'up' }
  ];

  // charts
  gasTrendChart: Partial<ApexOptions>;
  tempTrendChart: Partial<ApexOptions>;

  availableGases: string[] = ['NO2', 'O3', 'SO2', 'CO', 'CH4'];
  selectedGasKey = 'NO2';
  selectedGasStats: GasStats = { mean: 0, min: 0, max: 0 };

  constructor(
    private gasesService: AtmosphericGasesService,
    private router: Router
  ) {
    // weekly gas trend
    this.gasTrendChart = {
      chart: {
        type: 'line',
        height: 260,
        toolbar: { show: false },
        zoom: { enabled: false }
      },
      series: [{ name: 'Gas concentration', data: [] }],
      stroke: { curve: 'smooth', width: 3 },
      dataLabels: { enabled: false },
      xaxis: {
        categories: [],
        axisBorder: { show: false },
        axisTicks: { show: false }
      },
      yaxis: { title: { text: 'Concentration' } },
      colors: ['#4da3ff'],
      grid: { borderColor: '#e9edf5', strokeDashArray: 5 },
      legend: { position: 'top', horizontalAlign: 'right' }
    };

    // LST weekly trend
    this.tempTrendChart = {
      chart: {
        type: 'area',
        height: 260,
        toolbar: { show: false },
        zoom: { enabled: false }
      },
      series: [{ name: 'Land Surface Temp (°C)', data: [] }],
      dataLabels: { enabled: false },
      stroke: { curve: 'smooth', width: 2 },
      fill: {
        type: 'gradient',
        gradient: {
          shade: 'light',
          gradientToColors: ['#ff869a'],
          shadeIntensity: 0.9,
          opacityFrom: 0.9,
          opacityTo: 0.15,
          stops: [0, 90]
        }
      },
      colors: ['#ff6b81'],
      xaxis: {
        categories: [],
        axisBorder: { show: false },
        axisTicks: { show: false }
      },
      yaxis: { title: { text: '°C' } },
      grid: { borderColor: '#e9edf5', strokeDashArray: 5 },
      tooltip: {
        y: { formatter: (val: number) => `${val.toFixed(1)} °C` }
      }
    };
  }

  ngOnInit(): void {
    this.resolveUserNameFromJwt();
    this.loadRegion(this.selectedRegion);
  }

  // region / gas change
  onRegionChange(event: Event): void {
    const select = event.target as HTMLSelectElement | null;
    const region = select?.value ?? this.selectedRegion;
    this.selectedRegion = region;
    this.loadRegion(region);
  }

  onGasChange(event: Event): void {
    const select = event.target as HTMLSelectElement | null;
    const gas = (select?.value || this.selectedGasKey).toUpperCase();
    this.selectedGasKey = gas;
    this.updateGasTrendChartFromS5P();
  }

  // logout -> signin
  onLogout(event: MouseEvent): void {
    event.stopPropagation();
    this.userMenuOpen = false;

    ['token', 'access_token', 'jwt_token', 'auth_token', 'currentUser'].forEach(
      (k) => localStorage.removeItem(k)
    );

    this.router.navigate(['/auth/signin']);
  }

  // load data for region
  private loadRegion(region: string): void {
    this.isLoading = true;
    this.errorMessage = null;

    forkJoin({
      s5p: this.gasesService.getLatestS5P(region, undefined, 7),
      s3: this.gasesService.getLatestS3LST(region, 7)
    }).subscribe({
      next: ({ s5p, s3 }) => {
        this.latestS5p = s5p;
        this.latestS3 = s3;

        if (this.latestS5p?.gases?.length) {
          this.availableGases = this.latestS5p.gases.map((g) => g.toUpperCase());
        } else {
          this.availableGases = ['NO2', 'O3', 'SO2', 'CO', 'CH4'];
        }
        if (!this.availableGases.includes(this.selectedGasKey)) {
          this.selectedGasKey = this.availableGases[0];
        }

        this.applyBackendData();
        this.isLoading = false;
      },
      error: (err) => {
        console.error('Failed to load region data', err);
        this.errorMessage = `Could not load latest satellite data for ${region}.`;
        this.isLoading = false;
        this.latestS5p = null;
        this.latestS3 = null;
        this.applyBackendData();
      }
    });
  }

  private applyBackendData(): void {
    this.updateAQICardsFromS5P();
    this.updateLSTCardFromS3();
    this.updateGasTrendChartFromS5P();
    this.updateTempTrendChartFromS3();
  }

  // AQI cards for gases
  private updateAQICardsFromS5P(): void {
    const gasMeans: Record<string, number | null> = {};

    if (this.latestS5p) {
      for (const gas of this.latestS5p.gases) {
        const key = gas.toUpperCase();
        const docs = this.latestS5p.results[gas] || [];
        if (!docs.length) {
          gasMeans[key] = null;
          continue;
        }
        const latest = docs[0];
        const mean = latest.stats?.mean;
        gasMeans[key] =
          typeof mean === 'number' && !Number.isNaN(mean) ? mean : null;
      }
    }

    this.aqiCards = this.aqiCards.map((card) => {
      if (card.key === 'LST') return card;

      const key = card.key.toUpperCase();
      const value = gasMeans[key] ?? null;

      const updated: AQICard = { ...card, value };

      if (value === null) {
        updated.level = 'No data';
        updated.message =
          'No recent satellite data for this gas in this region (satellite or pipeline did not return values).';
      } else {
        updated.level = this.classifyAirQuality(key, value);
        updated.message = undefined;
      }
      return updated;
    });
  }

  // LST card
  private updateLSTCardFromS3(): void {
    const idx = this.aqiCards.findIndex((c) => c.key === 'LST');
    if (idx === -1) return;

    let value: number | null = null;

    if (this.latestS3 && this.latestS3.results.length) {
      const latest = this.latestS3.results[0];
      const mean = latest.stats?.mean;
      value = typeof mean === 'number' && !Number.isNaN(mean) ? mean : null;
    }

    const old = this.aqiCards[idx];
    const updated: AQICard = { ...old, value };

    if (value === null) {
      updated.level = 'No data';
      updated.message =
        'No recent land surface temperature data for this region (satellite or pipeline did not return values).';
    } else {
      updated.level = 'Info';
      updated.message = undefined;
    }

    const clone = [...this.aqiCards];
    clone[idx] = updated;
    this.aqiCards = clone;
  }

  private classifyAirQuality(gasKey: string, value: number): string {
    if (value <= 0) return 'No data';
    if (value < 30) return 'Good';
    if (value < 60) return 'Moderate';
    if (value < 90) return 'Unhealthy';
    return 'Very unhealthy';
  }

  // weekly gas trend + stats
  private updateGasTrendChartFromS5P(): void {
    if (!this.latestS5p) {
      this.gasTrendChart = {
        ...this.gasTrendChart,
        series: [{ name: 'Gas concentration', data: [] }],
        xaxis: { ...(this.gasTrendChart.xaxis || {}), categories: [] }
      };
      this.selectedGasStats = { mean: 0, min: 0, max: 0 };
      return;
    }

    const gasKey = this.selectedGasKey.toUpperCase();
    const docs =
      this.latestS5p.results[gasKey] ||
      this.latestS5p.results[gasKey.replace('NO2', 'NO₂')] ||
      [];

    if (!docs.length) {
      this.gasTrendChart = {
        ...this.gasTrendChart,
        series: [{ name: 'Gas concentration', data: [] }],
        xaxis: { ...(this.gasTrendChart.xaxis || {}), categories: [] }
      };
      this.selectedGasStats = { mean: 0, min: 0, max: 0 };
      return;
    }

    const categories: string[] = [];
    const values: number[] = [];

    for (const d of docs) {
      categories.push(d.date);
      const v = typeof d.stats?.mean === 'number' ? d.stats.mean : 0;
      values.push(v);
    }

    const valid = values.filter((v) => typeof v === 'number' && !isNaN(v));
    const mean =
      valid.length > 0 ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
    const min = valid.length ? Math.min(...valid) : 0;
    const max = valid.length ? Math.max(...valid) : 0;

    this.selectedGasStats = { mean, min, max };

    const meta = this.gasMeta[gasKey] || { label: gasKey, unit: '' };

    this.gasTrendChart = {
      ...this.gasTrendChart,
      series: [
        {
          name: `${meta.label} (${meta.unit})`,
          data: values
        }
      ],
      xaxis: { ...(this.gasTrendChart.xaxis || {}), categories },
      yaxis: { title: { text: `${meta.label} (${meta.unit})` } }
    };
  }

  // LST weekly trend chart
  private updateTempTrendChartFromS3(): void {
    if (!this.latestS3 || !this.latestS3.results.length) {
      this.tempTrendChart = {
        ...this.tempTrendChart,
        series: [{ name: 'Land Surface Temp (°C)', data: [] }],
        xaxis: { ...(this.tempTrendChart.xaxis || {}), categories: [] }
      };
      return;
    }

    const docs = this.latestS3.results;
    const categories = docs.map((d) => d.date);
    const data = docs.map((d) => d.stats?.mean ?? 0);

    this.tempTrendChart = {
      ...this.tempTrendChart,
      series: [{ name: 'Land Surface Temp (°C)', data }],
      xaxis: { ...(this.tempTrendChart.xaxis || {}), categories }
    };
  }

  // JWT → userName
  private resolveUserNameFromJwt(): void {
    const token = this.getStoredToken();
    if (!token) return;

    try {
      const payload = this.decodeJwtPayload(token);
      if (!payload) return;

      const fullName = (payload.full_name || '').trim();
      const username = (payload.username || payload.sub || '').trim();

      if (fullName) {
        this.userName = fullName.split(' ')[0];
      } else if (username) {
        this.userName = username;
      }
    } catch {
      // keep default
    }
  }

  private getStoredToken(): string | null {
    const keys = ['token', 'access_token', 'jwt_token', 'auth_token'];
    for (const k of keys) {
      const v = localStorage.getItem(k);
      if (v) return v;
    }
    return null;
  }

  private decodeJwtPayload(token: string): any | null {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    const base64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const json = atob(base64);
    return JSON.parse(json);
  }
}
