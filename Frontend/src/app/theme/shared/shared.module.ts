// src/app/theme/shared/shared.module.ts
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';

import {
  NgbDropdownModule,
  NgbProgressbarModule,
  NgbNavModule
} from '@ng-bootstrap/ng-bootstrap';

import { NgApexchartsModule } from 'ng-apexcharts';

import { CardComponent } from './card/card.component';

@NgModule({
  imports: [
    CommonModule,
    RouterModule,
    FormsModule,
    NgbDropdownModule,
    NgbProgressbarModule,
    NgbNavModule,
    NgApexchartsModule,
    CardComponent            // standalone component
  ],
  exports: [
    CommonModule,
    RouterModule,
    FormsModule,
    NgbDropdownModule,
    NgbProgressbarModule,
    NgbNavModule,
    NgApexchartsModule,
    CardComponent
  ]
})
export class SharedModule {}
