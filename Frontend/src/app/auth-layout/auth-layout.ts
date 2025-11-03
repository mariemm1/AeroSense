import {
  Component, Inject, PLATFORM_ID, AfterViewInit,
  ElementRef, ViewChild
} from '@angular/core';
import { NgIf, isPlatformBrowser } from '@angular/common';

@Component({
  selector: 'auth-layout',
  standalone: true,
  imports: [NgIf],
  templateUrl: './auth-layout.html',
  styleUrls: ['./auth-layout.css'],
})
export class AuthLayout implements AfterViewInit {
  y = new Date().getFullYear();

  @ViewChild('canvasRef', { static: false })
  canvasRef!: ElementRef<HTMLDivElement>;

  private isBrowser = false;

  constructor(@Inject(PLATFORM_ID) platformId: Object) {
    this.isBrowser = isPlatformBrowser(platformId);
  }

  async ngAfterViewInit() {
    if (!this.isBrowser) return;

    const host = this.canvasRef?.nativeElement;
    if (!host) return;

    // Load JS only (no CSS here)
    const L = await import('leaflet');
    // @ts-ignore (we have a local shim)
    await import('leaflet.heat');

    // Ensure host has size before map creation
    host.style.position = 'absolute';
    host.style.inset = '0';
    host.style.width = '100%';
    host.style.height = '100%';
    host.style.minHeight = '420px';
    host.classList.add('leaflet-container');

    const map = L.map(host as unknown as HTMLElement, {
      zoomControl: false,
      attributionControl: false,
    }).setView([36.8, 10.2], 6);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap contributors',
    }).addTo(map);

    const points: [number, number, number?][] = [
      [36.8065, 10.1815, 0.9], [36.451, 10.735, 0.7], [35.8256, 10.636, 0.6],
      [35.7643, 10.8113, 0.5], [35.6781, 10.0963, 0.55], [34.737, 10.756, 0.65],
      [33.8869, 9.5375, 0.4],  [33.8, 10.85, 0.45],   [33.139, 11.219, 0.5],
      [32.9297, 10.4518, 0.6],
    ];
    (L as any).heatLayer(points, {
      radius: 22, blur: 16, maxZoom: 11,
      gradient: { 0.2:'#3b82f6', 0.4:'#22c55e', 0.6:'#eab308', 0.8:'#f97316', 1:'#ef4444' }
    }).addTo(map);

    setTimeout(() => map.invalidateSize(), 0);
    window.addEventListener('resize', () => map.invalidateSize());
  }
}
