import { Component, Input, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SharedModule } from '../../theme/shared/shared.module';
import { ForecastResponse, AQIResponse } from '../../models/atmospheric-gases.model';

@Component({
  selector: 'app-forecast-aqi',
  standalone: true,
  imports: [CommonModule, SharedModule],
  templateUrl: './forecast-aqi.html',
  styleUrls: ['./forecast-aqi.css'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ForecastAqi {
  @Input() forecast: ForecastResponse | null = null;
  @Input() aqi: AQIResponse | null = null;

  get aqiColor(): string {
    if (!this.aqi) return 'neutral';
    switch (this.aqi.class_name) {
      case 'Good': return 'good';
      case 'Moderate': return 'moderate';
      case 'Unhealthy': return 'bad';
      default: return 'neutral';
    }
  }

  get forecastAqiColor(): string {
  if (!this.forecast || !this.forecast.aqi_class) return 'neutral';
  switch (this.forecast.aqi_class) {
    case 'Good': return 'good';
    case 'Moderate': return 'moderate';
    case 'Unhealthy': return 'bad';
    default: return 'neutral';
  }
}

}
