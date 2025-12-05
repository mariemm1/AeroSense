// Shared dashboard types

export interface RegionOption {
  id: string;    // backend region key (must match REGION_BBOXES)
  label: string; // human-readable label
}

export interface AQICard {
  pollutant: string;
  key: string;            // "NO2", "O3", ...
  value: number | null;
  unit: string;
  level: string;          // "Good", "Moderate", ...
  bgClass: string;        // css class for gradient background
  message?: string;
}

export interface SummaryCard {
  title: string;
  value: string;
  label: string;
  trend: 'up' | 'down' | 'neutral';
}

export interface GasStats {
  mean: number;
  min: number;
  max: number;
}

// All Tunisian regions (must match REGION_BBOXES in FastAPI backend)
export const TUNISIA_REGIONS: RegionOption[] = [
  // Whole country
  { id: 'tunisia',     label: 'Tunisia (country)' },

  // Northern & coastal governorates
  { id: 'tunis',       label: 'Tunis' },
  { id: 'ariana',      label: 'Ariana' },
  { id: 'ben_arous',   label: 'Ben Arous' },
  { id: 'manouba',     label: 'Manouba' },
  { id: 'bizerte',     label: 'Bizerte' },
  { id: 'nabeul',      label: 'Nabeul' },
  { id: 'zaghouan',    label: 'Zaghouan' },
  { id: 'beja',        label: 'Béja' },
  { id: 'jendouba',    label: 'Jendouba' },
  { id: 'kef',         label: 'Le Kef' },
  { id: 'siliana',     label: 'Siliana' },

  // Centre
  { id: 'kairouan',    label: 'Kairouan' },
  { id: 'sousse',      label: 'Sousse' },
  { id: 'monastir',    label: 'Monastir' },
  { id: 'mahdia',      label: 'Mahdia' },
  { id: 'sidi_bouzid', label: 'Sidi Bouzid' },
  { id: 'kasserine',   label: 'Kasserine' },
  { id: 'gafsa',       label: 'Gafsa' },

  // South & Sahara
  { id: 'sfax',        label: 'Sfax' },
  { id: 'gabes',       label: 'Gabès' },
  { id: 'medenine',    label: 'Medenine' },
  { id: 'tataouine',   label: 'Tataouine' },
  { id: 'kebili',      label: 'Kebili' },
  { id: 'tozeur',      label: 'Tozeur' }
];
