import { Routes } from '@angular/router';
import { HomePage } from './home-page/home-page';
import { Signup } from './signup/signup';
import { Signin } from './signin/signin';
import { Dashboard } from './dashboard/dashboard/dashboard';
import { ContactComponent } from './contact/contact.component';
import { AuthGuard } from './guards/auth-guard-guard';
import { AboutComponent } from './aboutUs/about.component';

export const routes: Routes = [
  // Default: go to home page
  { path: '', redirectTo: 'homPage', pathMatch: 'full' },

  // Auth
  { path: 'signin', component: Signin },
  { path: 'signup', component: Signup },

  // Main pages
  { path: 'homPage', component: HomePage },
  { path: 'dashboard', component: Dashboard, canActivate: [AuthGuard] },
  { path: 'contact', component: ContactComponent },
  {
  path: 'about', component: AboutComponent },



  // Wildcard – back to home page (fixed: 'home' → 'homPage')
  { path: '**', redirectTo: 'homPage' },
];
