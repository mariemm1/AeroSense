import {
  Component,
  Input,
  Output,
  EventEmitter,
  ChangeDetectionStrategy
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgApexchartsModule, ApexOptions } from 'ng-apexcharts';
import { SharedModule } from '../../theme/shared/shared.module';
import { GasStats } from '../../models/dashboard.types';

@Component({
  selector: 'app-gas-trend',
  standalone: true,
  imports: [CommonModule, SharedModule, NgApexchartsModule],
  templateUrl: './gas-trend.html',
  styleUrls: ['./gas-trend.css'],
  changeDetection: ChangeDetectionStrategy.OnPush
})

export class GasTrend {

  @Input() regionId = '';
  @Input() chart!: Partial<ApexOptions>;
  @Input() selectedGasKey = 'NO2';
  @Input() availableGases: string[] = [];
  @Input() selectedGasStats!: GasStats;
  @Input() gasMeta: Record<string, { label: string; unit: string }> = {};

  @Output() gasChange = new EventEmitter<string>();

  onGasChange(event: Event): void {
    const select = event.target as HTMLSelectElement | null;
    const gas = (select?.value || this.selectedGasKey).toUpperCase();
    this.gasChange.emit(gas);
  }

}
