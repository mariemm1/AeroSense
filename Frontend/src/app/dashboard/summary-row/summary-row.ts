import { Component, Input, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SharedModule } from '../../theme/shared/shared.module';
import { SummaryCard } from '../../models/dashboard.types';

@Component({
  selector: 'app-summary-row',
  standalone: true,
  imports: [CommonModule, SharedModule],
  templateUrl: './summary-row.html',
  styleUrls: ['./summary-row.css'],
  changeDetection: ChangeDetectionStrategy.OnPush
})

export class SummaryRow {

  @Input() cards: SummaryCard[] = [];

}
