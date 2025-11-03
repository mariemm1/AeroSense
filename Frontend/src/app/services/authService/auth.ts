import { Injectable, Inject, PLATFORM_ID } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { JwtHelperService } from '@auth0/angular-jwt';
import { Observable } from 'rxjs';
import { Router } from '@angular/router';

type SigninResp = {
  id: string; username: string; email: string; role: string;
  token?: string; token_expires?: string; needs_verification?: boolean; error?: string;
};

@Injectable({ providedIn: 'root' })
export class Auth {
  /** One place to change API base; no environments. */
  private API_BASE = 'http://127.0.0.1:8000';

  private token = '';
  private helper = new JwtHelperService();

  public isLoggedIn = false;
  public roles?: string;
  public loggedUsername?: string;

  private orgId?: string;
  private orgName?: string;

  constructor(
    private http: HttpClient,
    private router: Router,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  // ---------- SSR-safe storage helpers ----------
  private get hasBrowser(): boolean {
    return isPlatformBrowser(this.platformId) && typeof localStorage !== 'undefined';
  }
  private lsGet(key: string): string | null { return this.hasBrowser ? localStorage.getItem(key) : null; }
  private lsSet(key: string, val: string): void { if (this.hasBrowser) localStorage.setItem(key, val); }
  private lsRemove(key: string): void { if (this.hasBrowser) localStorage.removeItem(key); }

  // ---------- API ----------
  signup(data: {
    full_name: string; what_he_does: string; region: string;
    email: string; password: string; role?: string;
  }): Observable<any> {
    return this.http.post(`${this.API_BASE}/auth/signup/`, data);
  }

  signin(data: { username: string; password: string }): Observable<SigninResp> {
    return this.http.post<SigninResp>(`${this.API_BASE}/auth/signin/`, data);
  }

  resendVerification(login: string) {
    const body = login.includes('@') ? { email: login } : { username: login };
    return this.http.post(`${this.API_BASE}/auth/resend-verification/`, body);
  }

  // ---------- Token handling ----------
  saveToken(token: string): void {
    this.token = token;
    this.isLoggedIn = true;
    this.lsSet('jwt', token);
    this.lsSet('isLoggedIn', 'true');

    this.decodeJWT();

    const userId = this.getUserIdFromToken();
    if (userId)              this.lsSet('userId', userId);
    if (this.roles)          this.lsSet('role', this.roles);
    if (this.orgId)          this.lsSet('orgId', this.orgId);
    if (this.orgName)        this.lsSet('orgName', this.orgName);
    if (this.loggedUsername) this.lsSet('username', this.loggedUsername);

    // Also drop a cookie for SSR reads on first paint (safe defaults)
    if (this.hasBrowser) {
      const decoded: any = this.helper.decodeToken(token) || {};
      const nowSec = Math.floor(Date.now() / 1000);
      const maxAge = typeof decoded?.exp === 'number'
        ? Math.max(0, decoded.exp - nowSec)
        : 7 * 24 * 60 * 60;
      document.cookie = `jwt=${token}; Path=/; SameSite=Lax; Max-Age=${maxAge}`;
    }
  }

  loadToken(): void {
    const stored = this.lsGet('jwt');
    if (stored) {
      this.token = stored;
      this.decodeJWT();
      this.isLoggedIn = !this.helper.isTokenExpired(stored);
      this.orgId   = this.orgId   || this.lsGet('orgId')   || undefined;
      this.orgName = this.orgName || this.lsGet('orgName') || undefined;
      this.roles   = this.roles   || this.lsGet('role')    || (undefined as any);
      this.loggedUsername = this.loggedUsername || this.lsGet('username') || undefined;
    }
  }

  private decodeJWT(): void {
    if (!this.token) return;
    const decoded: any = this.helper.decodeToken(this.token) || {};
    this.loggedUsername = decoded?.username ?? decoded?.sub ?? undefined;
    this.roles = decoded?.role;

    this.orgId =
      decoded.organizationId ?? decoded.orgId ?? decoded.tenantId ?? decoded.org ?? this.orgId;

    const nameClaim =
      decoded.organizationName ?? decoded.orgName ?? decoded.tenantName ?? decoded.organization ?? decoded.org ?? null;
    this.orgName = typeof nameClaim === 'string' ? nameClaim : this.orgName;
  }

  logout(): void {
    const org = this.orgId || this.lsGet('orgId') || '';
    this.token = ''; this.roles = undefined; this.loggedUsername = undefined;
    this.orgId = undefined; this.orgName = undefined; this.isLoggedIn = false;

    this.lsRemove('jwt'); this.lsRemove('isLoggedIn'); this.lsRemove('userId');
    this.lsRemove('role'); this.lsRemove('orgId'); this.lsRemove('orgName'); this.lsRemove('username');

    if (this.hasBrowser) document.cookie = 'jwt=; Path=/; Max-Age=0; SameSite=Lax';

    this.router.navigate(['/signin', org]);
  }

  // ---------- Helpers ----------
  getToken(): string { if (!this.token) this.token = this.lsGet('jwt') || ''; return this.token; }
  isAuthenticated(): boolean { const t = this.getToken(); return !!t && !this.helper.isTokenExpired(t); }
  isTokenExpired(): boolean { return this.helper.isTokenExpired(this.getToken()); }
  getRoleFromToken(): string | null { return this.roles || this.lsGet('role'); }
  getUserIdFromToken(): string { const d: any = this.helper.decodeToken(this.getToken()) || {}; return d?.id?.toString?.() || d?.sub || ''; }
  getUsername(): string | null { return this.loggedUsername || this.lsGet('username'); }
  getHeaders(): HttpHeaders { return new HttpHeaders({ Authorization: `Bearer ${this.getToken()}` }); }
  getOrganizationFromToken(): string { return this.orgId || this.lsGet('orgId') || ''; }
  getOrganizationNameFromToken(): string { return this.orgName || this.lsGet('orgName') || ''; }
}
