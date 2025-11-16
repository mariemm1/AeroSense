// src/app/theme/shared/card/card.component.ts

import { CommonModule } from '@angular/common';
import { Component, Input, OnInit, inject, input } from '@angular/core';
import { animate, AUTO_STYLE, state, style, transition, trigger } from '@angular/animations';

// ng-bootstrap
import { NgbDropdownConfig, NgbDropdownModule } from '@ng-bootstrap/ng-bootstrap';

@Component({
  selector: 'app-card', 
  standalone: true,
  imports: [CommonModule, NgbDropdownModule],
  templateUrl: './card.component.html',
  styleUrls: ['./card.component.css'],
  providers: [NgbDropdownConfig],
  animations: [
    // collapse animation for card body
    trigger('collapsedCard', [
      state(
        'collapsed, void',
        style({
          overflow: 'hidden',
          height: '0px'
        })
      ),
      state(
        'expanded',
        style({
          overflow: 'hidden',
          height: AUTO_STYLE
        })
      ),
      transition('collapsed <=> expanded', [animate('400ms ease-in-out')])
    ]),
    // fade-out animation when card is removed
    trigger('cardRemove', [
      state(
        'open',
        style({
          opacity: 1
        })
      ),
      state(
        'closed',
        style({
          opacity: 0,
          display: 'none'
        })
      ),
      transition('open <=> closed', animate('400ms'))
    ])
  ]
})
export class CardComponent implements OnInit {
  // -------- public inputs (same logic as your template) --------
  @Input() cardTitle = 'Card Title';
  @Input() cardClass = '';      // extra classes for outer card
  @Input() options = true;      // show / hide options menu
  @Input() hidHeader = false;   // hide card header
  @Input() customHeader = false;// use <ng-content> for header
  @Input() customDate = false;  // use <ng-content> for subheader/date
  @Input() isCardFooter = false;// show footer area

  // signal-based inputs (so you can still use headerClass(), blockClass() etc.)
  blockClass = input<string>('');   // [ngClass]="blockClass()"
  headerClass = input<string>('');  // [ngClass]="headerClass()"
  footerClass = input<string>('');  // [ngClass]="footerClass()"
  CardDate = input<string>('');     // {{ CardDate() }}

  // -------- internal state --------
  animation = '';
  fullIcon = 'icon-maximize';
  isAnimating = false;

  collapsedCard: 'collapsed' | 'expanded' | 'false' = 'expanded';
  collapsedIcon = 'icon-minus';

  loadCard = false;
  cardRemove: 'open' | 'closed' = 'open';

  constructor() {
    // configure dropdown placement
    const config = inject(NgbDropdownConfig) as NgbDropdownConfig;
    config.placement = 'bottom-right';
  }

  ngOnInit(): void {
    // when header or options are disabled, don't animate collapse
    if (!this.options || this.hidHeader || this.customHeader || this.customDate) {
      this.collapsedCard = 'false';
    }
  }

  // -------- public methods used in template --------

  /** Toggle full-screen card */
  fullCardToggle(element: HTMLElement, animation: string, status: boolean): void {
    animation = this.cardClass === 'full-card' ? 'zoomOut' : 'zoomIn';
    this.fullIcon = this.cardClass === 'full-card' ? 'icon-maximize' : 'icon-minimize';
    this.cardClass = this.cardClass === 'full-card' ? '' : 'full-card';

    if (status) {
      this.animation = animation;
    }
    this.isAnimating = true;

    setTimeout(() => {
      if (animation === 'zoomOut') {
        this.cardClass = '';
      }

      const body = document.querySelector('body') as HTMLBodyElement | null;
      if (!body) return;

      if (this.cardClass === 'full-card') {
        body.style.overflow = 'hidden';
      } else {
        body.removeAttribute('style');
      }
    }, 500);
  }

  /** Collapse / expand card body */
  collapsedCardToggle(): void {
    this.collapsedCard = this.collapsedCard === 'collapsed' ? 'expanded' : 'collapsed';
    this.collapsedIcon = this.collapsedCard === 'collapsed' ? 'icon-plus' : 'icon-minus';
  }

  /** Fake "reload" animation */
  cardRefresh(): void {
    this.loadCard = true;
    this.cardClass = 'card-load';
    setTimeout(() => {
      this.loadCard = false;
      this.cardClass = 'expanded';
    }, 3000);
  }

  /** Remove the card from view */
  cardRemoveAction(): void {
    this.cardRemove = this.cardRemove === 'closed' ? 'open' : 'closed';
  }
}
