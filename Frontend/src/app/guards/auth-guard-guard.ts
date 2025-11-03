import { Injectable } from '@angular/core';
import { CanActivate, Router, ActivatedRouteSnapshot, RouterStateSnapshot, UrlTree } from '@angular/router';
import { Auth } from '../services/authService/auth';

@Injectable({ providedIn: 'root' })
export class AuthGuard implements CanActivate {
  constructor(private auth: Auth, private router: Router) {}

  canActivate(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): boolean | UrlTree {
    this.auth.loadToken(); // SSR-safe hydrate
    if (this.auth.isAuthenticated()) return true;

    const orgId = this.auth.getOrganizationFromToken() || '';
    return this.router.createUrlTree(['/signin', orgId], { queryParams: { returnUrl: state.url } });
  }
}
