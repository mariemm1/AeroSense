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
liveStats = [
    { gas: 'CO', value: '0.12', unit: 'mol/m²', level: 'faible', color: '#28a745' },
    { gas: 'NO₂', value: '45', unit: 'µg/m³', level: 'moyen', color: '#ffc107' },
    { gas: 'O₃', value: '92', unit: 'µg/m³', level: 'élevé', color: '#dc3545' },
    { gas: 'CH₄', value: '1910', unit: 'ppb', level: 'faible', color: '#28a745' },
    { gas: 'SO₂', value: '15', unit: 'µg/m³', level: 'faible', color: '#28a745' },
    { gas: 'LST', value: '22.5', unit: '°C', level: 'moyen', color: '#ffc107' }
  ];

  features = [
    { icon: 'satellite_alt', title: 'Données Sentinel-5P', desc: 'Acquisition automatique' },
    { icon: 'psychology', title: 'Deep Learning', desc: 'Prédiction et classification' },
    { icon: 'notifications_active', title: 'Alertes AQI', desc: 'Messages personnalisés' },
    { icon: 'cloud_queue', title: 'Temps réel', desc: 'Visualisation instantanée' }
  ];

}
