import {
  ChangeDetectionStrategy,
  Component,
  OnInit
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { forkJoin } from 'rxjs';

import { AtmosphericGasesService } from '../../services/atmospheric-gases/atmospheric-gases';
import {
  LatestS3LSTResponse,
  LatestS5PResponse,
  ForecastResponse,
  AQIResponse
} from '../../models/atmospheric-gases.model';
import {
  AQICard,
  SummaryCard,
  GasStats,
  RegionOption,
  TUNISIA_REGIONS
} from '../../models/dashboard.types';

import { AqiCards } from '../aqi-cards/aqi-cards';
import { SummaryRow } from '../summary-row/summary-row';
import { GasTrend } from '../gas-trend/gas-trend';
import { LstTrend } from '../lst-trend/lst-trend';
import { RegionMap } from '../region-map/region-map';
import { ForecastAqi } from '../forecast-aqi/forecast-aqi';
import { ApexOptions } from 'ng-apexcharts';

import { Auth } from '../../services/authService/auth';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    AqiCards,
    SummaryRow,
    GasTrend,
    LstTrend,
    RegionMap,
    ForecastAqi
  ],
  templateUrl: './dashboard.html',
  styleUrls: ['./dashboard.css'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class Dashboard implements OnInit {
  // REGIONS – all Tunisian regions (backend REGION_BBOXES)
  regions: RegionOption[] = TUNISIA_REGIONS;
  selectedRegion = 'tunisia';

  // STATE
  isLoading = false;
  errorMessage: string | null = null;

  latestS5p: LatestS5PResponse | null = null;
  latestS3: LatestS3LSTResponse | null = null;
  forecast: ForecastResponse | null = null;
  aqi: AQIResponse | null = null;

  // ALL-REGION AQI FOR MAP
  aqiRegions: AQIResponse[] | null = null;

  // CURRENT USER
  userName: string | null = null;
  isUserMenuOpen = false;

  // GAS META
  readonly gasMeta: Record<string, { label: string; unit: string }> = {
    NO2: { label: 'NO₂', unit: 'µmol/m²' },
    O3: { label: 'O₃', unit: 'µg/m³' },
    SO2: { label: 'SO₂', unit: 'µg/m³' },
    CO:  { label: 'CO',  unit: 'mg/m³' },
    CH4: { label: 'CH₄', unit: 'ppb' },
    LST: { label: 'LST', unit: '°C' }
  };

  // AQI MINI-CARDS
  aqiCards: AQICard[] = [
    { pollutant: 'NO₂', key: 'NO2', value: null, unit: this.gasMeta['NO2'].unit, level: 'No data', bgClass: 'aqi-bg-no2' },
    { pollutant: 'O₃', key: 'O3', value: null, unit: this.gasMeta['O3'].unit, level: 'No data', bgClass: 'aqi-bg-o3' },
    { pollutant: 'SO₂', key: 'SO2', value: null, unit: this.gasMeta['SO2'].unit, level: 'No data', bgClass: 'aqi-bg-so2' },
    { pollutant: 'CO',  key: 'CO',  value: null, unit: this.gasMeta['CO'].unit,  level: 'No data', bgClass: 'aqi-bg-co' },
    { pollutant: 'CH₄', key: 'CH4', value: null, unit: this.gasMeta['CH4'].unit, level: 'No data', bgClass: 'aqi-bg-ch4' },
    { pollutant: 'LST', key: 'LST', value: null, unit: this.gasMeta['LST'].unit, level: 'No data', bgClass: 'aqi-bg-lst' }
  ];

  // SUMMARY KPI CARDS
  summaryCards: SummaryCard[] = [
    { title: 'Daily AQI',        value: '—', label: 'ML classification',   trend: 'neutral' },
    { title: 'Weekly Avg AQI',   value: '—', label: 'From satellite data', trend: 'neutral' },
    { title: 'Monthly Avg AQI',  value: '—', label: 'Coming soon',         trend: 'neutral' },
    { title: 'Coverage',         value: '—', label: 'Grid cells / regions',trend: 'neutral' }
  ];

  // CHARTS
  gasTrendChart: Partial<ApexOptions>;
  tempTrendChart: Partial<ApexOptions>;

  availableGases: string[] = ['NO2', 'O3', 'SO2', 'CO', 'CH4'];
  selectedGasKey = 'NO2';
  selectedGasStats: GasStats = { mean: 0, min: 0, max: 0 };

  constructor(
    private gasesService: AtmosphericGasesService,
    private auth: Auth
  ) {
    this.gasTrendChart = {
      chart: {
        type: 'line',
        height: 260,
        toolbar: { show: false },
        zoom: { enabled: false },
        foreColor: '#c3d4ff',
        background: 'transparent'
      },
      series: [{ name: 'Gas concentration', data: [] }],
      stroke: { curve: 'smooth', width: 3 },
      dataLabels: { enabled: false },
      xaxis: {
        categories: [],
        axisBorder: { show: false },
        axisTicks: { show: false },
        labels: { style: { colors: '#9fb2dd' } }
      },
      yaxis: {
        title: { text: 'Concentration', style: { color: '#9fb2dd' } },
        labels: { style: { colors: '#9fb2dd' } }
      },
      colors: ['#4da3ff'],
      grid: { borderColor: '#202b46', strokeDashArray: 4 },
      legend: {
        position: 'top',
        horizontalAlign: 'right',
        labels: { colors: '#e6eeff' }
      }
    };

    this.tempTrendChart = {
      chart: {
        type: 'area',
        height: 260,
        toolbar: { show: false },
        zoom: { enabled: false },
        foreColor: '#c3d4ff',
        background: 'transparent'
      },
      series: [{ name: 'Land Surface Temp (°C)', data: [] }],
      dataLabels: { enabled: false },
      stroke: { curve: 'smooth', width: 2 },
      fill: {
        type: 'gradient',
        gradient: {
          shade: 'dark',
          gradientToColors: ['#ff869a'],
          shadeIntensity: 0.9,
          opacityFrom: 0.9,
          opacityTo: 0.12,
          stops: [0, 90]
        }
      },
      colors: ['#ff6b81'],
      xaxis: {
        categories: [],
        axisBorder: { show: false },
        axisTicks: { show: false },
        labels: { style: { colors: '#9fb2dd' } }
      },
      yaxis: {
        title: { text: '°C', style: { color: '#9fb2dd' } },
        labels: { style: { colors: '#9fb2dd' } }
      },
      grid: { borderColor: '#202b46', strokeDashArray: 4 },
      tooltip: {
        y: { formatter: (val: number) => `${val.toFixed(1)} °C` }
      }
    };
  }

  ngOnInit(): void {
    this.userName = this.auth.getUsername();
    this.loadRegion(this.selectedRegion);
    this.loadAllRegionsAQI();      // <-- load AQI for ALL regions for the map
  }

  // REGION CHANGE FROM TOPBAR
  onRegionChange(event: Event): void {
    const select = event.target as HTMLSelectElement | null;
    const region = select?.value ?? this.selectedRegion;
    this.selectedRegion = region;
    this.loadRegion(region);
  }

  // USER MENU
  toggleUserMenu(): void {
    this.isUserMenuOpen = !this.isUserMenuOpen;
  }

  // LOGOUT
  onLogout(): void {
    this.isUserMenuOpen = false;
    this.auth.logout();
  }

  // GAS CHANGE FROM CHILD
  onGasSelectionChange(gasKey: string): void {
    this.selectedGasKey = gasKey.toUpperCase();
    this.updateGasTrendChartFromS5P();
  }

  // ---------- LOAD AQI FOR ALL REGIONS (for the global map) ----------
  private loadAllRegionsAQI(): void {
    const regionIds = this.regions.map(r => r.id);
    const requests = regionIds.map(id => this.gasesService.getAQI(id));

    forkJoin(requests).subscribe({
      next: (results) => {
        // keep only truthy results
        this.aqiRegions = results.filter((r): r is AQIResponse => !!r);
      },
      error: (err) => {
        console.error('Failed to load AQI for all regions', err);
        this.aqiRegions = null;
      }
    });
  }

  // ---------- DATA LOADING FOR SELECTED REGION ----------

  private loadRegion(region: string): void {
    this.isLoading = true;
    this.errorMessage = null;

    forkJoin({
      s5p: this.gasesService.getLatestS5P(region, undefined, 7),
      s3: this.gasesService.getLatestS3LST(region, 7),
      forecast: this.gasesService.getForecast(region),
      aqi: this.gasesService.getAQI(region)
    }).subscribe({
      next: ({ s5p, s3, forecast, aqi }) => {
        this.latestS5p = s5p;
        this.latestS3 = s3;
        this.forecast = forecast;
        this.aqi = aqi;

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
        this.forecast = null;
        this.aqi = null;

        this.applyBackendData();
      }
    });
  }

  private applyBackendData(): void {
    this.updateAQICardsFromS5P();
    this.updateLSTCardFromS3();
    this.updateGasTrendChartFromS5P();
    this.updateTempTrendChartFromS3();
    this.updateSummaryFromAQI();
  }

  // --- AQI CARDS ---
  private updateAQICardsFromS5P(): void {
    const gasMeans: Record<string, number | null> = {};

    if (this.latestS5p) {
      for (const gas of this.latestS5p.gases) {
        const key = gas.toUpperCase();
        const docs = this.latestS5p.results[gas] || this.latestS5p.results[key] || [];
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

  private classifyAirQuality(_gasKey: string, value: number): string {
    if (value <= 0 || Number.isNaN(value)) return 'No data';
    if (value < 30) return 'Good';
    if (value < 60) return 'Moderate';
    if (value < 90) return 'Unhealthy';
    return 'Very unhealthy';
  }

  // --- CHARTS ---
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
    const docsRaw =
      this.latestS5p.results[gasKey] ||
      this.latestS5p.results[gasKey.toUpperCase()] ||
      this.latestS5p.results[gasKey.toLowerCase()] ||
      [];

    if (!docsRaw.length) {
      this.gasTrendChart = {
        ...this.gasTrendChart,
        series: [{ name: 'Gas concentration', data: [] }],
        xaxis: { ...(this.gasTrendChart.xaxis || {}), categories: [] }
      };
      this.selectedGasStats = { mean: 0, min: 0, max: 0 };
      return;
    }

    const docs = [...docsRaw].reverse();
    const categories: string[] = [];
    const values: number[] = [];

    for (const d of docs) {
      categories.push(d.date);
      const v =
        typeof d.stats?.mean === 'number' && !Number.isNaN(d.stats.mean)
          ? d.stats.mean
          : 0;
      values.push(v);
    }

    const valid = values.filter((v) => typeof v === 'number' && !Number.isNaN(v));
    const mean = valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
    const min  = valid.length ? Math.min(...valid) : 0;
    const max  = valid.length ? Math.max(...valid) : 0;

    this.selectedGasStats = { mean, min, max };

    this.gasTrendChart = {
      ...this.gasTrendChart,
      series: [{ name: `${gasKey} mean`, data: values }],
      xaxis: { ...(this.gasTrendChart.xaxis || {}), categories }
    };
  }

  private updateTempTrendChartFromS3(): void {
    if (!this.latestS3 || !this.latestS3.results.length) {
      this.tempTrendChart = {
        ...this.tempTrendChart,
        series: [{ name: 'Land Surface Temp (°C)', data: [] }],
        xaxis: { ...(this.tempTrendChart.xaxis || {}), categories: [] }
      };
      return;
    }

    const docs = [...this.latestS3.results].reverse();
    const categories: string[] = [];
    const values: number[] = [];

    for (const d of docs) {
      categories.push(d.date);
      const v =
        typeof d.stats?.mean === 'number' && !Number.isNaN(d.stats.mean)
          ? d.stats.mean
          : 0;
      values.push(v);
    }

    this.tempTrendChart = {
      ...this.tempTrendChart,
      series: [{ name: 'Land Surface Temp (°C)', data: values }],
      xaxis: { ...(this.tempTrendChart.xaxis || {}), categories }
    };
  }

  // --- SUMMARY KPI FROM AQI / COVERAGE ---
  private updateSummaryFromAQI(): void {
    const dailyClass = this.aqi?.class_name ?? 'N/A';
    const coverage = this.latestS5p?.top ?? 0;
    const regions = this.latestS5p?.gases?.length ?? 0;

    this.summaryCards = [
      {
        title: 'Daily AQI',
        value: dailyClass,
        label: this.aqi ? `for ${this.aqi.region} (${this.aqi.date})` : 'ML classification',
        trend: this.aqi
          ? (this.aqi.class_id === 0 ? 'up'
            : this.aqi.class_id === 1 ? 'neutral'
            : 'down')
          : 'neutral'
      },
      {
        title: 'Weekly Avg AQI',
        value: this.selectedGasStats.mean ? this.selectedGasStats.mean.toFixed(1) : '—',
        label: 'Mean of selected gas (last 7 days)',
        trend: 'neutral'
      },
      {
        title: 'Monthly Avg AQI',
        value: '—',
        label: 'Coming soon',
        trend: 'neutral'
      },
      {
        title: 'Coverage',
        value: coverage && regions ? `${coverage} / ${regions}` : '—',
        label: 'Grid cells / gases',
        trend: 'up'
      }
    ];
  }
}
