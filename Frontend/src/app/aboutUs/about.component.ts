import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-about',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './about.component.html',
  styleUrls: ['./about.component.css'],
})
export class AboutComponent {
  gases = [
    {
      code: 'NO₂',
      name: 'Dioxyde d’azote',
      role: 'Polluant des zones urbaines',
      text:
        'Issu principalement du trafic routier et des centrales thermiques, le NO₂ irrite les voies respiratoires et contribue à la formation de l’ozone troposphérique.',
      badge: 'Qualité de l’air urbain',
    },
    {
      code: 'O₃',
      name: 'Ozone troposphérique',
      role: 'Polluant secondaire',
      text:
        'Formé à partir des NOₓ et COV sous l’effet du soleil. À basse altitude, l’ozone provoque des irritations et impacte les cultures agricoles.',
      badge: 'Smog photochimique',
    },
    {
      code: 'CO',
      name: 'Monoxyde de carbone',
      role: 'Gaz toxique',
      text:
        'Produit par la combustion incomplète (chauffage, trafic, incendies). Il réduit la capacité du sang à transporter l’oxygène.',
      badge: 'Combustion incomplète',
    },
    {
      code: 'SO₂',
      name: 'Dioxyde de soufre',
      role: 'Polluant industriel',
      text:
        'Provenant surtout de la combustion de carburants soufrés et de certaines industries. Il est à l’origine des pluies acides et de fortes irritations.',
      badge: 'Industrie & énergie',
    },
    {
      code: 'CH₄',
      name: 'Méthane',
      role: 'Gaz à effet de serre puissant',
      text:
        'Émis par l’agriculture, l’élevage, les fuites de gaz et les zones humides. Son pouvoir de réchauffement est bien plus élevé que celui du CO₂ à court terme.',
      badge: 'Climat & émissions diffuses',
    },
    {
      code: 'LST',
      name: 'Land Surface Temperature',
      role: 'Température de la surface',
      text:
        'Mesure la température de la surface terrestre. Indispensable pour suivre les vagues de chaleur, les îlots de chaleur urbains et le stress hydrique des sols.',
      badge: 'Chaleur & stress hydrique',
    },
  ];

  // 🔻 "Alertes intelligentes" entry removed
  models = [
    {
      title: 'Classification AQI multi-classes',
      text:
        'Nous utilisons des modèles de Deep Learning pour classer la qualité de l’air (bonne, modérée, mauvaise…) à partir des observations Sentinel-5P et de variables météorologiques.',
    },
    {
      title: 'Prévision temporelle des gaz',
      text:
        'Des réseaux récurrents et d’autres architectures séquentielles apprennent l’évolution des concentrations de NO₂, O₃, SO₂, CO et CH₄ afin d’anticiper les épisodes de pollution.',
    },
    {
      title: 'Analyse conjointe gaz + LST',
      text:
        'En combinant Land Surface Temperature et gaz atmosphériques, nous détectons les zones à risque : chaleur extrême, pollution persistante, stress pour les populations et les cultures.',
    },
  ];

  workflowSteps = [
    {
      step: '01',
      title: 'Acquisition des données',
      text:
        'Collecte automatisée des produits Sentinel-5P (NO₂, O₃, SO₂, CO, CH₄) et des données de température de surface, complétées par la météo et les informations au sol.',
    },
    {
      step: '02',
      title: 'Pré-traitement & normalisation',
      text:
        'Filtrage spatial sur la Tunisie, agrégation par régions, interpolation temporelle, calcul d’indicateurs dérivés et normalisation des séries temporelles.',
    },
    {
      step: '03',
      title: 'Modélisation & apprentissage',
      text:
        'Entraînement des modèles de classification AQI et de prévision sur plusieurs années de données, avec validation croisée et optimisation des hyperparamètres.',
    },
    {
      step: '04',
      title: 'Visualisation & alertes',
      text:
        'Les résultats sont projetés sur des cartes interactives, résumés sur des dashboards et exposés via API pour être intégrés dans d’autres systèmes.',
    },
  ];

  audiences = [
    {
      title: 'Collectivités & autorités publiques',
      text:
        'Suivi en temps réel de la qualité de l’air, soutien à la décision pour les plans d’action, communication transparente envers les citoyens.',
    },
    {
      title: 'Chercheurs & universités',
      text:
        'Accès à des séries temporelles harmonisées, tests de modèles IA et études d’impact sur la santé, le climat ou les écosystèmes.',
    },
    {
      title: 'Industrie & énergie',
      text:
        'Surveillance des émissions autour des sites sensibles, mise en place d’indicateurs ESG et suivi de l’empreinte environnementale.',
    },
    {
      title: 'Agriculture & smart irrigation',
      text:
        'Croisement LST / gaz / météo pour mieux comprendre le stress hydrique, optimiser l’irrigation et protéger les cultures lors des épisodes de pollution.',
    },
  ];
}
