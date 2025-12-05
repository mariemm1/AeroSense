import {
  Component,
  Input,
  ChangeDetectionStrategy,
  AfterViewInit,
  OnChanges,
  OnDestroy,
  SimpleChanges,
  ViewChild,
  ElementRef,
  Inject,
  PLATFORM_ID
} from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { SharedModule } from '../../theme/shared/shared.module';
import { AQIResponse } from '../../models/atmospheric-gases.model';

@Component({
  selector: 'app-region-map',
  standalone: true,
  imports: [CommonModule, SharedModule],
  templateUrl: './region-map.html',
  styleUrls: ['./region-map.css'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class RegionMap implements AfterViewInit, OnChanges, OnDestroy {
  // Single region (for caption fallback)
  @Input() aqi: AQIResponse | null = null;

  // ALL regions for the map
  @Input() aqiRegions: AQIResponse[] | null = null;

  // currently selected region id (e.g. "tunis", "sfax", "tunisia")
  @Input() selectedRegionId: string | null = null;

  @ViewChild('mapContainer', { static: false })
  mapContainer!: ElementRef<HTMLDivElement>;

  // Leaflet module (loaded dynamically in browser only)
  private L: typeof import('leaflet') | null = null;

  // Map + layer refs
  private map: any = null;
  private markerLayer: any = null;

  constructor(@Inject(PLATFORM_ID) private platformId: Object) {}

  private get isBrowser(): boolean {
    return isPlatformBrowser(this.platformId);
  }

  /**
   * Normalise region names to make matching robust:
   * - lowercase
   * - remove "governorate", "gouvernorat", "region", "city", "(country)" etc.
   * - replace -, _ with spaces and collapse multiple spaces
   */
  private normalizeName(name: string | null | undefined): string {
    if (!name) return '';
    let n = name.toLowerCase();

    n = n
      .replace(/\(country\)/g, '')
      .replace(/\(city\)/g, '')
      .replace(/governorate/g, '')
      .replace(/gouvernorat/g, '')
      .replace(/region/g, '')
      .replace(/city/g, '');

    n = n.replace(/[-_]/g, ' ');
    n = n.replace(/\s+/g, ' ').trim();
    return n;
  }

  // approximate centers for Tunisia + all 24 governorates
  private regionCoords: Record<string, [number, number]> = {
    // country
    tunisia: [34.0, 9.0],

    // Grand Tunis
    tunis: [36.8065, 10.1815],
    ariana: [36.8665, 10.1647],
    'ben arous': [36.7435, 10.231],
    manouba: [36.8081, 10.097],

    // North / Cap Bon
    nabeul: [36.4514, 10.7353],
    zaghouan: [36.4029, 10.1429],
    bizerte: [37.2746, 9.8739],
    beja: [36.725, 9.1817],
    jendouba: [36.5011, 8.7792],
    kef: [36.1742, 8.704],
    siliana: [36.0883, 9.3708],

    // Centre
    kairouan: [35.6781, 10.0963],
    kasserine: [35.1676, 8.8365],
    'sidi bouzid': [35.04, 9.484],

    // Sahel
    sousse: [35.8256, 10.6411],
    monastir: [35.7643, 10.8113],
    mahdia: [35.5047, 11.0622],

    // Centre / South-East coast
    sfax: [34.7406, 10.7603],
    gabes: [33.8815, 10.0982],
    medenine: [33.3549, 10.5055],
    tataouine: [32.9297, 10.4518],

    // South / inland
    gafsa: [34.425, 8.7842],
    tozeur: [33.9197, 8.1335],
    kebili: [33.7044, 8.969]
  };

  // ---------------- LIFECYCLE ----------------

  ngAfterViewInit(): void {
    if (!this.isBrowser) return;
    this.setupMap();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (!this.isBrowser) return;

    if (
      (changes['aqi'] ||
        changes['aqiRegions'] ||
        changes['selectedRegionId']) &&
      this.map
    ) {
      this.updateMarkers();
    }
  }

  ngOnDestroy(): void {
    if (this.map) {
      this.map.remove();
      this.map = null;
      this.markerLayer = null;
    }
  }

  // ---------------- LEAFLET LOADING ----------------

  private async setupMap(): Promise<void> {
    if (!this.isBrowser) return;

    if (!this.L) {
      const leaflet = await import('leaflet'); // dynamic import – SSR safe
      this.L = leaflet;
    }

    if (!this.map) {
      this.initMap();
    }
    // Draw markers immediately with whatever data we already have
    this.updateMarkers();
  }

  // ---------------- MAP SETUP ----------------

  private initMap(): void {
    if (!this.mapContainer || !this.L) return;

    const L = this.L;

    this.map = L.map(this.mapContainer.nativeElement, {
      center: [34.0, 9.0], // Tunisia centre
      zoom: 6,
      minZoom: 4,
      maxZoom: 10
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(this.map);

    this.markerLayer = L.layerGroup().addTo(this.map);
  }

  private getCoordsForRegion(name?: string | null): [number, number] {
    if (!name) return [34.0, 9.0];

    const key = this.normalizeName(name);

    // 1) direct match after normalisation
    if (this.regionCoords[key]) {
      return this.regionCoords[key];
    }

    // 2) fuzzy: substring match (handles "tunis governorate", "sfax-city" etc.)
    for (const [k, coords] of Object.entries(this.regionCoords)) {
      if (key.includes(k)) {
        return coords;
      }
    }

    // fallback: country centre
    return [34.0, 9.0];
  }

  private getColorForClassId(classId: number | null | undefined): string {
    if (classId === 0) return '#34c759'; // Good
    if (classId === 1) return '#ffcc00'; // Moderate
    if (classId === 2) return '#ff3b30'; // Unhealthy
    return '#9fa4b5'; // No data / default
  }

  private isSelectedRegion(regionAQI: AQIResponse): boolean {
    if (!this.selectedRegionId) return false;

    const selected = this.normalizeName(this.selectedRegionId);
    const region = this.normalizeName(regionAQI.region);

    if (!selected || !region) return false;

    return selected === region || selected.includes(region) || region.includes(selected);
  }

  private updateMarkers(): void {
    if (!this.map || !this.markerLayer || !this.L) return;
    const L = this.L;

    this.markerLayer.clearLayers();

    // Prefer all-region data; fall back to single region if needed
    const regionsArray: AQIResponse[] =
      this.aqiRegions && this.aqiRegions.length
        ? this.aqiRegions
        : this.aqi
        ? [this.aqi]
        : [];

    if (!regionsArray.length) {
      this.map.setView([34.0, 9.0], 6);
      return;
    }

    const bounds: [number, number][] = [];

    for (const regionAQI of regionsArray) {
      const coords = this.getCoordsForRegion(regionAQI.region);
      const color = this.getColorForClassId(regionAQI.class_id);
      const selected = this.isSelectedRegion(regionAQI);

      const circle = L.circleMarker(coords, {
        radius: selected ? 11 : 7,
        color: selected ? '#ffffff' : color,
        fillColor: color,
        fillOpacity: selected ? 1 : 0.65,
        weight: selected ? 3 : 1.5,
        className: selected ? 'aqi-marker selected-region' : 'aqi-marker'
      });

      const popupHtml = `
        <div class="aqi-popup">
          <strong>${regionAQI.region}</strong><br/>
          Date: ${regionAQI.date}<br/>
          Class: ${regionAQI.class_name}<br/>
          Good: ${(regionAQI.probabilities['0'] * 100).toFixed(0)}%<br/>
          Moderate: ${(regionAQI.probabilities['1'] * 100).toFixed(0)}%<br/>
          Unhealthy: ${(regionAQI.probabilities['2'] * 100).toFixed(0)}%
        </div>
      `;

      circle.bindPopup(popupHtml);
      circle.addTo(this.markerLayer);
      bounds.push(coords);
    }

    if (bounds.length === 1) {
      this.map.setView(bounds[0], 7);
    } else {
      this.map.fitBounds(bounds as any, { padding: [40, 40] });
    }
  }
}
