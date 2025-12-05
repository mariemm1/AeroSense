import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-home-page',
  imports: [CommonModule, RouterLink],
  templateUrl: './home-page.html',
  styleUrl: './home-page.css',
})
export class HomePage {
  // Only keep what is actually used on the page
  features = [
    { icon: 'satellite_alt', title: 'Données Sentinel-5P', desc: 'Acquisition automatique' },
    { icon: 'psychology', title: 'Deep Learning', desc: 'Prédiction et classification' },
    { icon: 'notifications_active', title: 'Alertes AQI', desc: 'Messages personnalisés' },
    { icon: 'cloud_queue', title: 'Temps réel', desc: 'Visualisation instantanée' }
  ];
}
