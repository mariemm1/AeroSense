import { Component, Input, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgApexchartsModule, ApexOptions } from 'ng-apexcharts';
import { SharedModule } from '../../theme/shared/shared.module';

@Component({
  selector: 'app-lst-trend',
  standalone: true,
  imports: [CommonModule, SharedModule, NgApexchartsModule],
  templateUrl: './lst-trend.html',
  styleUrls: ['./lst-trend.css'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class LstTrend {
  @Input() regionId = '';
  @Input() chart!: Partial<ApexOptions>;
}
