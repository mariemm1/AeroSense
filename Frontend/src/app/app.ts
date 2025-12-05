import { Component, inject, signal } from '@angular/core';
import { Router, RouterOutlet, NavigationEnd } from '@angular/router';
import { NgIf } from '@angular/common';
import { filter } from 'rxjs/operators';

import { Navbar } from './navbar/navbar';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, Navbar, NgIf],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected readonly title = signal('Frontend');

  private readonly router = inject(Router);
  readonly showNavbar = signal(true);

  constructor() {
    // initial state
    this.updateNavbarVisibility(this.router.url);

    // update on route change
    this.router.events
      .pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd))
      .subscribe((e: NavigationEnd) => {
        const url = e.urlAfterRedirects || e.url;
        this.updateNavbarVisibility(url);
      });
  }

  private updateNavbarVisibility(url: string): void {
    const cleanUrl = url.split('?')[0].split('#')[0];
    const isDashboard = cleanUrl.startsWith('/dashboard');
    this.showNavbar.set(!isDashboard);
  }
}
