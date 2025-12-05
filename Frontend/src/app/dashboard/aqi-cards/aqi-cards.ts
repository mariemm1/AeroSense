import { Component, Input, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SharedModule } from '../../theme/shared/shared.module';
import { AQICard } from '../../models/dashboard.types';

@Component({
  selector: 'app-aqi-cards',
  standalone: true,
  imports: [CommonModule, SharedModule],
  templateUrl: './aqi-cards.html',
  styleUrls: ['./aqi-cards.css'],
  changeDetection: ChangeDetectionStrategy.OnPush
})


export class AqiCards {

  @Input() cards: AQICard[] = [];

}
